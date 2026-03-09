"""
Generate a pre-trimmed context JSON file for a given model and target token size.

Takes a master context file (~1M tokens of math or text distractor Q&A),
trims it to an exact token count using the model's native tokenizer, and
saves the result as a ready-to-use context JSON file.

Output filename format:
    context_{type}_{size}_{safe_model_name}.json

Example:
    python generate_context_file.py \
        --model tiiuae/Falcon-H1R-7B \
        --context_type math \
        --context_size 16000
"""

import argparse
import json
import os
import re
import sys

from transformers import AutoTokenizer
from util import BASELINE_SYSTEM_PROMPT


def strip_thinking_tags(text):
    """Remove <think>, <thinking>, etc. tags that may appear in assistant responses."""
    return re.sub(
        r'</?(think|thinking|reasoning|reflection|scratchpad)[^>]*>',
        '', text, flags=re.IGNORECASE
    ).strip()


def format_size(n_tokens):
    """Convert token count to human-readable string: 16000 -> '16K', 750000 -> '750K'."""
    if n_tokens >= 1000 and n_tokens % 1000 == 0:
        return f"{n_tokens // 1000}K"
    return str(n_tokens)


def generate_trimmed_context(tokenizer, messages, target_size):
    """
    Binary search to find the exact number of messages that fits within target_size tokens.
    Phase 1: message-level binary search.
    Phase 2: word-level truncation of the next message to fill remaining capacity.
    """
    # Strip existing system prompts (we'll prepend our own)
    messages = [m for m in messages if m['role'] != 'system']

    # Strip thinking tags from all assistant messages
    for msg in messages:
        msg['content'] = strip_thinking_tags(msg['content'])

    system_msg = {"role": "system", "content": BASELINE_SYSTEM_PROMPT}

    def count_tokens(msg_list):
        rendered = tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=False)
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    # Phase 1: Binary search on full messages
    low = 0
    high = len(messages)
    best_k = 0

    print(f"  Phase 1: Binary search over {len(messages)} messages for {target_size:,} tokens...")

    while low <= high:
        mid = (low + high) // 2
        test_chunk = [system_msg] + messages[:mid]
        n_tokens = count_tokens(test_chunk)

        # Sanity check: tokenizer should produce real output
        if mid > 0 and n_tokens <= 2:
            raise ValueError(
                f"Tokenizer returned only {n_tokens} tokens for {mid} messages. "
                f"This usually means the model's chat_template is missing or broken."
            )

        if n_tokens <= target_size:
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1

    result_msgs = [system_msg] + messages[:best_k]
    current_tokens = count_tokens(result_msgs)
    print(f"  Phase 1 result: {best_k} messages, {current_tokens:,} tokens")

    # Phase 2: Word-level truncation of the next message
    if current_tokens < target_size and best_k < len(messages):
        gap = target_size - current_tokens
        print(f"  Phase 2: Filling remaining ~{gap:,} tokens with word-level truncation...")
        next_msg = messages[best_k]
        words = next_msg['content'].split()

        low_w = 0
        high_w = len(words)
        best_w = 0

        while low_w <= high_w:
            mid_w = (low_w + high_w) // 2
            partial_content = " ".join(words[:mid_w])
            test_chunk = result_msgs + [{"role": next_msg['role'], "content": partial_content}]
            n_tokens = count_tokens(test_chunk)

            if n_tokens <= target_size:
                best_w = mid_w
                low_w = mid_w + 1
            else:
                high_w = mid_w - 1

        if best_w > 0:
            result_msgs.append({"role": next_msg['role'], "content": " ".join(words[:best_w])})

    final_tokens = count_tokens(result_msgs)
    print(f"  Final: {len(result_msgs)} messages, {final_tokens:,} tokens (target: {target_size:,})")

    return result_msgs, final_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Generate a pre-trimmed context JSON file for local model evaluation."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model ID (used to load the correct tokenizer)")
    parser.add_argument("--context_type", type=str, choices=["math", "text"], required=True,
                        help="Type of context to trim ('math' or 'text')")
    parser.add_argument("--context_size", type=int, required=True,
                        help="Target token count for the output context file")
    parser.add_argument("--master_file", type=str, default=None,
                        help="Path to master context JSON. Defaults to "
                             "context_saturation/contexts/context_{type}_1M.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Defaults to context_saturation/contexts/")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code when loading tokenizer")
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.master_file is None:
        args.master_file = os.path.join(
            base_dir, "context_saturation", "contexts",
            f"context_{args.context_type}_1M.json"
        )

    if args.output_dir is None:
        args.output_dir = os.path.join(base_dir, "context_saturation", "contexts")

    if not os.path.exists(args.master_file):
        print(f"Error: Master context file not found: {args.master_file}")
        sys.exit(1)

    # 1. Load tokenizer
    print(f"Loading tokenizer for: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}, vocab size: {tokenizer.vocab_size}")

    if tokenizer.chat_template is None:
        print(f"WARNING: Model '{args.model}' has no chat_template. Results may be unreliable.")

    # 2. Load master file
    print(f"\nLoading master context: {args.master_file}")
    with open(args.master_file, 'r') as f:
        master_messages = json.load(f)
    print(f"  {len(master_messages)} messages loaded")

    # 3. Trim
    print(f"\nTrimming to {args.context_size:,} tokens...")
    trimmed_messages, actual_tokens = generate_trimmed_context(
        tokenizer, master_messages, args.context_size
    )

    # 4. Save
    safe_model = args.model.replace('/', '_')
    size_str = format_size(args.context_size)
    filename = f"context_{args.context_type}_{size_str}_{safe_model}.json"

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, filename)

    with open(out_path, 'w') as f:
        json.dump(trimmed_messages, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"  Messages: {len(trimmed_messages)}")
    print(f"  Tokens:   {actual_tokens:,} (target: {args.context_size:,})")


if __name__ == "__main__":
    main()
