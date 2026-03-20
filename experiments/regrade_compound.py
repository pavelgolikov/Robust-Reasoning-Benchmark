#!/usr/bin/env python3
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import extract_and_grade

def main():
    parser = argparse.ArgumentParser(description='Re-grade a single compound result file')
    parser.add_argument('filepath', type=str, help='Path to the JSON file to regrade')
    parser.add_argument('--apply', action='store_true', help='Actually overwrite files (default: dry run)')
    parser.add_argument('--model', type=str, default=None, help='Exact model name or path for tokenizer')
    args = parser.parse_args()

    filepath = args.filepath
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        results = data
        is_list = True
    elif isinstance(data, dict) and 'results' in data:
        results = data['results']
        is_list = False
    else:
        print("Unrecognized format.")
        return

    # Check if there's a summary block at the end of a list of results (evaluate.py format)
    summary_idx = -1
    for i, r in enumerate(results):
        if 'summary' in r:
            summary_idx = i
            break
            
    summary_block = None
    max_tokens = 32768
    model_name = args.model
    if summary_idx != -1:
        # separate it
        summary_block = results.pop(summary_idx)
        max_tokens = summary_block['summary'].get('max_tokens', summary_block['summary'].get('max_model_length', 32768))
        model_name = summary_block['summary'].get('model', None)

    if not model_name:
        # Try to infer from filepath: .../results/model_name/dataset/...
        parts = filepath.replace('\\', '/').split('/')
        try:
            results_idx = parts.index('results')
            model_name = parts[results_idx + 1]
        except (ValueError, IndexError):
            raise RuntimeError("Could not determine model name to load exact tokenizer from JSON.")
            
        # The folder name replaces slashes with underscores, try to undo it for HF models
        if "_" in model_name and not model_name.startswith("gpt") and not model_name.startswith("claude") and not model_name.startswith("gemini"):
            model_name = model_name.replace('_', '/', 1)

    print(f"Loading exact tokenizer for model {model_name}...")
    def get_exact_token_counter(m_name):
        lower_model = m_name.lower()
        if 'gpt' in lower_model or 'o1' in lower_model:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(m_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return lambda text: len(enc.encode(text))
        elif 'claude' in lower_model:
            import anthropic
            client = anthropic.Anthropic()
            return lambda text: client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=m_name,
                messages=[{"role": "user", "content": text}]
            ).input_tokens
        elif 'gemini' in lower_model:
            from google import genai
            client = genai.Client()
            return lambda text: client.models.count_tokens(model=m_name, contents=text).total_tokens
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(m_name, trust_remote_code=True)
            return lambda text: len(tokenizer.encode(text))

    token_counter = get_exact_token_counter(model_name)

    changed = 0
    total = len(results)
    flips_correct = 0
    flips_incorrect = 0
    max_token_cutoffs = 0

    print(f"Regrading {total} entries...")

    for r in results:
        output = r.get('output', '')
        gt = r.get('ground_truth', '')
        old_correct = r.get('correct', False)
        old_extracted = r.get('extracted', None)

        new_extracted, new_correct = extract_and_grade(output, gt, exp_name='compound')

        if new_correct != old_correct or str(new_extracted) != str(old_extracted):
            changed += 1
            direction = ""
            if not old_correct and new_correct:
                direction = "  ✓ FIXED"
                flips_correct += 1
            elif old_correct and not new_correct:
                direction = "  ✗ REGRESSED"
                flips_incorrect += 1
            else:
                direction = "  ~ extracted changed"
                
            print(f"ID {r.get('id', '?')}s{r.get('sample_idx', 0)}: "
                  f"\"{old_extracted}\" -> \"{new_extracted}\" "
                  f"({old_correct} -> {new_correct}) {direction}")
                  
            r['extracted'] = str(new_extracted) if new_extracted is not None else None
            r['correct'] = new_correct

        # Calculate exact token count
        try:
            token_count = token_counter(output)
        except Exception as e:
            raise RuntimeError(f"Failed to precisely count tokens using {model_name} tokenizer: {e}")
        
        is_cutoff = token_count >= max_tokens * 0.95
        if is_cutoff:
            max_token_cutoffs += 1

    # Restore summary block with updated stats if it existed
    if summary_block:
        new_correct_count = sum(1 for r in results if r.get('correct', False))
        new_failures = len(results) - new_correct_count
        summary_block['summary']['correct'] = new_correct_count
        summary_block['summary']['failures'] = new_failures
        summary_block['summary']['max_token_cutoffs'] = max_token_cutoffs
        accuracy = new_correct_count / len(results) if len(results) > 0 else 0
        summary_block['summary']['accuracy'] = accuracy
        results.append(summary_block)
        
    print(f"\n--- Summary ---")
    print(f"Items regraded: {total}")
    print(f"Changed items:  {changed}")
    print(f"Fixed:          {flips_correct}")
    print(f"Regressed:      {flips_incorrect}")
    print(f"Max token cut:  {max_token_cutoffs}")

    if summary_block:
        print(f"New Accuracy:   {accuracy:.2%} ({new_correct_count}/{total})")

    if args.apply and changed > 0:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print("Updated file saved.")
    elif changed > 0:
        print("Run with --apply to save changes.")

if __name__ == '__main__':
    main()
