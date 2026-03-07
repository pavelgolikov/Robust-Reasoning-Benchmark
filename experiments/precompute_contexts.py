import os
import sys
import json
import argparse
from transformers import AutoTokenizer

from trim_context import trim_context

def main():
    parser = argparse.ArgumentParser(description="Pre-calculate trimmed context files to save time.")
    parser.add_argument("--sizes", type=str, default="750000,500000,250000", help="Comma-separated target token sizes.")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Model to use for tokenizer.")
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    context_files = ["context_math_1M.json", "context_text_1M.json"]

    # Initialize tokenizer
    print(f"Loading tokenizer for model: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load specific tokenizer ({e}). Falling back to gpt2.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if not getattr(tokenizer, 'chat_template', None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] + ': ' + message['content'] + '\n' }}"
            "{% endfor %}"
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for filename in context_files:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}, skipping.")
            continue
        
        # Get base name without the `_1M` suffix if it exists, for cleaner output names
        # e.g., context_math_1M.json -> context_math_750K.json
        base_name = filename.replace("_1M.json", "").replace(".json", "")

        for size in sizes:
            size_k = f"{size // 1000}K"
            out_filename = f"{base_name}_{size_k}.json"
            out_path = os.path.join(base_dir, out_filename)

            if os.path.exists(out_path):
                print(f"File {out_filename} already exists. Skipping.")
                continue

            print(f"\nProcessing {filename} -> {out_filename} ({size} tokens)...")
            try:
                trimmed = trim_context(path, args.model, size, tokenizer=tokenizer)
                with open(out_path, 'w') as f:
                    json.dump(trimmed, f, indent=2)
                print(f"Successfully saved {out_filename}.")
            except Exception as e:
                print(f"Error trimming {filename} to {size}: {e}")

if __name__ == "__main__":
    main()
