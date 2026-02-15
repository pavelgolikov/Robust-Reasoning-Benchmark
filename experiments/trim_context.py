import argparse
import json
import os
import sys

# Ensure local imports work whether run from root or experiments/
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def count_tokens(tokenizer, history):
    full_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(full_text))

def trim_message_content(tokenizer, content, target_reduction):
    """
    Trims content from the end to reduce token count by approximately target_reduction.
    """
    # Simply encode, slice, decode
    ids = tokenizer.encode(content, add_special_tokens=False)
    current_len = len(ids)
    
    if current_len <= target_reduction:
        return "" # Remove entire content
        
    keep_len = current_len - target_reduction
    # Keep strictly less than current to ensure reduction
    if keep_len >= current_len: keep_len = current_len - 1
    
    sliced_ids = ids[:keep_len]
    return tokenizer.decode(sliced_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Trim JSON context to exact token count")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON context file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON context file")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Model name for tokenizer")
    parser.add_argument("--token_target", type=int, required=True, help="Exact target token count")
    
    args = parser.parse_args()
    
    # Load Tokenizer
    print(f"Loading tokenizer for {args.model} via vLLM...")
    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load tokenizer: {e}")
        print("Ensure 'vllm' is installed and the model name is correct.")
        sys.exit(1)

    # Load Context
    with open(args.input_file, 'r') as f:
        full_history = json.load(f)
        
    # Check total
    total_tokens = count_tokens(tokenizer, full_history)
    print(f"Input Tokens: {total_tokens}")
    
    if total_tokens < args.token_target:
        print(f"Input is already strictly smaller than target {args.token_target}. Exiting.")
        exit(1)

    # Add messages until limit
    trimmed_history = []
    current_tokens = 0
    
    # Always keep system prompt if present first
    for i, msg in enumerate(full_history):
        # Temp add
        test_history = trimmed_history + [msg]
        test_count = count_tokens(tokenizer, test_history)
        
        if test_count <= args.token_target:
            trimmed_history.append(msg)
            current_tokens = test_count
            if current_tokens == args.token_target:
                break
        else:
            # Overflow!
            print(f"Message {i} causes overflow ({test_count} > {args.token_target}). Trimming...")
            content = msg['content']
            # Heuristic:
            overflow = test_count - args.token_target
            
            # Simple direct cut
            msg_ids = tokenizer.encode(content, add_special_tokens=False)
            current_msg_len = len(msg_ids)
            allowed = current_msg_len - overflow
            
            # Safety checks
            if allowed < 0: allowed = 0
            
            truncated_ids = msg_ids[:allowed]
            new_content = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            
            msg['content'] = new_content
            trimmed_history.append(msg)
            
            # Final verification (optional)
            test_history = trimmed_history
            current_tokens = count_tokens(tokenizer, test_history)
            print(f"Trimmed to {current_tokens} tokens.")
            
            break
            
    print(f"Final Token Count: {current_tokens}")
    
    with open(args.output_file, 'w') as f:
        json.dump(trimmed_history, f, indent=2)
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
