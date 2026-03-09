import argparse
import json
import os
import sys

# Ensure vLLM internal utilities are available
try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    print("Error: vLLM is not installed. Please install vllm to use this script.")
    sys.exit(1)

from util import BASELINE_SYSTEM_PROMPT

def generate_trimmed_context(tokenizer, messages, target_size):
    """
    Binary search to find exact message count for target_size.
    Uses the model's native chat template and tokenizer.
    """
    # Strip any existing system prompts
    messages = [m for m in messages if m['role'] != 'system']
    
    system_msg = {"role": "system", "content": BASELINE_SYSTEM_PROMPT}
    
    # Phase 1: Binary search on full messages
    low = 0
    high = len(messages)
    best_k = 0
    
    print(f"Phase 1: Binary search on {len(messages)} full messages for {target_size} tokens...")
    
    while low <= high:
        mid = (low + high) // 2
        test_chunk = [system_msg] + messages[:mid]
        
        # Standard two-step tokenization (Render -> Encode)
        # This is the most robust way to handle potentially buggy tokenize=True paths
        try:
            rendered = tokenizer.apply_chat_template(test_chunk, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer.encode(rendered, add_special_tokens=False)
            count = len(tokens)
        except Exception as e:
            print(f"Error applying chat template: {e}")
            raise

        if (mid > 0 or len(system_msg['content']) > 0) and count <= 2:
             raise ValueError(
                 f"Tokenizer failure: returned only {count} tokens for {mid} messages. "
                 "This often means the model's 'tokenizer_config.json' on the Hub is missing a 'chat_template'."
             )

        if count <= target_size:
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1

    result_msgs = [system_msg] + messages[:best_k]
    rendered_final = tokenizer.apply_chat_template(result_msgs, tokenize=False, add_generation_prompt=False)
    current_tokens = len(tokenizer.encode(rendered_final, add_special_tokens=False))
    
    # Phase 2: Word-level truncation
    if current_tokens < target_size and best_k < len(messages):
        print(f"Phase 2: Filling remaining {target_size - current_tokens} tokens with word-level truncation...")
        next_msg = messages[best_k]
        words = next_msg['content'].split()
        
        low_w = 0
        high_w = len(words)
        best_w = 0
        
        while low_w <= high_w:
            mid_w = (low_w + high_w) // 2
            partial_content = " ".join(words[:mid_w])
            test_chunk = result_msgs + [{"role": next_msg['role'], "content": partial_content}]
            rendered_test = tokenizer.apply_chat_template(test_chunk, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer.encode(rendered_test, add_special_tokens=False)
            
            if len(tokens) <= target_size:
                best_w = mid_w
                low_w = mid_w + 1
            else:
                high_w = mid_w - 1
        
        if best_w > 0:
            result_msgs.append({"role": next_msg['role'], "content": " ".join(words[:best_w])})

    final_rendered = tokenizer.apply_chat_template(result_msgs, tokenize=False, add_generation_prompt=False)
    final_count = len(tokenizer.encode(final_rendered, add_special_tokens=False))
    print(f"Final context: {len(result_msgs)} messages, {final_count} tokens.")
    
    return result_msgs

def main():
    parser = argparse.ArgumentParser(description="Generate pre-trimmed context JSON using vLLM's internal tokenizer.")
    parser.add_argument("--model", type=str, required=True, help="Model ID on HuggingFace Hub")
    parser.add_argument("--context_type", type=str, choices=['math', 'text'], required=True, help="Type of context to use")
    parser.add_argument("--context_size", type=int, required=True, help="Target token count")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the JSON output")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code when loading tokenizer")
    args = parser.parse_args()

    print(f"Loading vLLM-aligned tokenizer for: {args.model}")
    # Use vLLM's internal utility for perfect alignment
    tokenizer = get_tokenizer(args.model, trust_remote_code=args.trust_remote_code)

    if tokenizer.chat_template is None:
        raise ValueError(f"Model '{args.model}' is missing a 'chat_template' in its configuration. Please check the Hub.")

    # Load master file from the standard experiment location
    base_path = "/home/golikovp/Antigravity/Linguistic_traps/experiments/context_saturation/contexts"
    file_name = f"context_{args.context_type}_1M.json"
    input_path = os.path.join(base_path, file_name)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Master context file not found: {input_path}")

    print(f"Loading master context: {input_path}")
    with open(input_path, 'r') as f:
        master_messages = json.load(f)

    # Perform trimming
    trimmed_messages = generate_trimmed_context(tokenizer, master_messages, args.context_size)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(trimmed_messages, f, indent=2)
    print(f"Successfully saved trimmed context to: {args.output_file}")

if __name__ == "__main__":
    main()
