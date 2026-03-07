import argparse
import json
import os
import sys
# from vllm.transformers_utils.tokenizer import get_tokenizer # Moved inside function

# Ensure local imports work whether run from root or experiments/
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def count_tokens(tokenizer, history):
    full_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(full_text))

# def trim_message_content(tokenizer, content, target_reduction):
#     """
#     Trims content from the end to reduce token count by approximately target_reduction.
#     """
#     # Simply encode, slice, decode
#     ids = tokenizer.encode(content, add_special_tokens=False)
#     current_len = len(ids)
    
#     if current_len <= target_reduction:
#         return "" # Remove entire content
        
#     keep_len = current_len - target_reduction
#     # Keep strictly less than current to ensure reduction
#     if keep_len >= current_len: keep_len = current_len - 1
    
#     sliced_ids = ids[:keep_len]
#     return tokenizer.decode(sliced_ids, skip_special_tokens=True)

def trim_context(input_file, model, token_target, tokenizer=None):
    if tokenizer is None:
        try:
            from vllm.transformers_utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(model, trust_remote_code=True)
        except ImportError:
            print("vLLM not installed/found. Please provide a tokenizer or install vLLM.")
            # Fallback or exit? If dry run passed tokenizer, we are good. If not, we fail.
            sys.exit(1)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            sys.exit(1)

    # Load Context
    with open(input_file, 'r') as f:
        full_history = json.load(f)
        
    # Add messages until limit using Binary Search
    total_msgs = len(full_history)
    low = 1
    high = total_msgs
    k_best = 1
    
    print("Finding the right message index using binary search...")
    while low <= high:
        mid = (low + high) // 2
        test_count = count_tokens(tokenizer, full_history[:mid])
        if test_count <= token_target:
            k_best = mid
            low = mid + 1
        else:
            high = mid - 1
            
    trimmed_history = full_history[:k_best]
    current_tokens = count_tokens(tokenizer, trimmed_history)
    
    if current_tokens == token_target or k_best == total_msgs:
        return trimmed_history
        
    # The next message causes overflow. We must trim its content.
    overflow_msg = full_history[k_best].copy()
    test_history = trimmed_history + [overflow_msg]
    test_count = count_tokens(tokenizer, test_history)
    
    print(f"Message {k_best} causes overflow ({test_count} > {token_target}). Trimming...")
    content = overflow_msg['content']
    overflow = test_count - token_target
    
    msg_ids = tokenizer.encode(content, add_special_tokens=False)
    allowed = len(msg_ids) - overflow
    if allowed < 0: allowed = 0
    
    truncated_ids = msg_ids[:allowed]
    new_content = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    
    overflow_msg['content'] = new_content
    trimmed_history.append(overflow_msg)
    
    # Final verification
    final_tokens = count_tokens(tokenizer, trimmed_history)
    print(f"Trimmed to {final_tokens} tokens.")
            
    return trimmed_history

# if __name__ == "__main__":
#     main()
