import json
import os
import numpy as np
from transformers import AutoTokenizer

def analyze_file(filepath, tokenizer):
    print(f"Analyzing {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return

    prompt_lengths = []
    completion_lengths = []
    total_lengths = []

    # Iterate through the list. Assumes strict User -> Assistant -> User -> Assistant ordering
    # or just distinguishes by role.
    
    current_prompt_len = 0
    
    for entry in data:
        role = entry.get("role")
        content = entry.get("content", "")
        tokens = len(tokenizer.encode(content))
        
        if role == "user":
            current_prompt_len = tokens
        elif role == "assistant":
            prompt_lengths.append(current_prompt_len)
            completion_lengths.append(tokens)
            total_lengths.append(current_prompt_len + tokens)
            current_prompt_len = 0 # Reset

    if not total_lengths:
        print("No valid user-assistant pairs found.")
        return

    print(f"  Total Queries: {len(total_lengths)}")
    print(f"  Prompt Tokens:    Mean={np.mean(prompt_lengths):.1f}, Median={np.median(prompt_lengths):.1f}, Min={np.min(prompt_lengths)}, Max={np.max(prompt_lengths)}")
    print(f"  Completion Tokens: Mean={np.mean(completion_lengths):.1f}, Median={np.median(completion_lengths):.1f}, Min={np.min(completion_lengths)}, Max={np.max(completion_lengths)}")
    print(f"  Total per Query:  Mean={np.mean(total_lengths):.1f}, Median={np.median(total_lengths):.1f}, Min={np.min(total_lengths)}, Max={np.max(total_lengths)}")
    print("-" * 40)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to load a tokenizer. specific model might need access or be slow, so we can use gpt2 as a proxy if needed.
    model_name = "gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer {model_name}: {e}")
        return

    analyze_file(os.path.join(script_dir, "../context_math.json"), tokenizer)
    analyze_file(os.path.join(script_dir, "../context_text.json"), tokenizer)

if __name__ == "__main__":
    main()
