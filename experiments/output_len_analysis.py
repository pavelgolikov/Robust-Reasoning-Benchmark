import argparse
import json
import os
import glob
from transformers import AutoTokenizer

def get_model_name(filepath):
    normalized_path = os.path.normpath(filepath)
    parts = normalized_path.split(os.sep)
    if "results" in parts:
        try:
            results_idx = parts.index("results")
            # The folder right after 'results' is the safe_model_name
            safe_model_name = parts[results_idx + 1]
            # Convert safe_model_name back to HuggingFace org/model format
            return safe_model_name.replace('_', '/', 1)
        except IndexError:
            pass
    return None

def is_proprietary(model_name):
    lower_name = model_name.lower()
    basename = lower_name.split('/')[-1]
    # Check for common proprietary API model prefixes
    proprietary_prefixes = ['gpt-', 'claude-', 'gemini-', 'o1-', 'o3-']
    return any(p in lower_name for p in proprietary_prefixes) or any(basename.startswith(p) for p in proprietary_prefixes)

def process_file(json_file, tokenizers_cache):
    print(f"Processing {json_file}...")
    
    model_name = get_model_name(json_file)
    if not model_name:
        raise ValueError(f"Could not automatically determine model name from path {json_file}")
        
    if is_proprietary(model_name):
        print(f"Skipping proprietary model: {model_name} (token limits usually handled by API or not locally enumerable).")
        return
        
    if model_name not in tokenizers_cache:
        print(f"Loading exact tokenizer for model: {model_name}...")
        try:
            tokenizers_cache[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load fast tokenizer for {model_name} ({e}), trying use_fast=False...")
            try:
                tokenizers_cache[model_name] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            except Exception as e2:
                raise RuntimeError(f"ERROR: Correct tokenizer not found for {model_name}. Initial error: {e}. Slow error: {e2}")
            
    tokenizer = tokenizers_cache[model_name]
    
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    summary_idx = -1
    for i, item in enumerate(data):
        if 'summary' in item:
            summary_idx = i
            break
            
    if summary_idx == -1:
        print(f"No summary block found in {json_file}. Skipping.")
        return
        
    tokens_list = []
    
    # Process all entries before the summary block
    for i in range(len(data)):
        if i == summary_idx:
            continue
            
        entry = data[i]
        
        # Get purely the output text
        text = entry.get('output', '')
        
        # Calculate tokens
        tokens = len(tokenizer.encode(text))
        entry['output_tokens'] = tokens
        tokens_list.append(tokens)
        
    if tokens_list:
        avg_tokens = sum(tokens_list) / len(tokens_list)
        min_tokens = min(tokens_list)
        max_tokens = max(tokens_list)
        
        # Update summary block
        summary = data[summary_idx]['summary']
        summary['avg_output_tokens'] = avg_tokens
        summary['min_output_tokens'] = min_tokens
        summary['max_output_tokens'] = max_tokens
        
        # Write back to file
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"  Successfully updated summary:")
        print(f"    Avg tokens: {avg_tokens:.1f}")
        print(f"    Min tokens: {min_tokens}")
        print(f"    Max tokens: {max_tokens}")
    else:
        print(f"  No valid entries found to process.")

def main():
    parser = argparse.ArgumentParser(description="Analyze and update token counts using the exact model tokenizer.")
    parser.add_argument("path", help="Path to a JSON file or directory containing JSON files.")
    
    args = parser.parse_args()
    
    # Cache tokenizers so we don't reload them for every file
    tokenizers_cache = {}
    
    if os.path.isfile(args.path):
        if not args.path.endswith('_raw.json'):
            process_file(args.path, tokenizers_cache)
        else:
            print("Skipping raw checkpoint file.")
    elif os.path.isdir(args.path):
        json_files = glob.glob(os.path.join(args.path, "**/*.json"), recursive=True)
        for jf in json_files:
            if not jf.endswith('_raw.json'):
                process_file(jf, tokenizers_cache)
    else:
        print(f"Invalid path: {args.path}")

if __name__ == "__main__":
    main()
