#!/usr/bin/env python3
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import extract_and_grade

def get_exact_token_counter(model_name):
    lower_model = model_name.lower()
    if 'gpt' in lower_model or 'o1' in lower_model:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    elif 'claude' in lower_model:
        import anthropic
        client = anthropic.Anthropic()
        return lambda text: client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model=model_name,
            messages=[{"role": "user", "content": text}]
        ).input_tokens
    elif 'gemini' in lower_model:
        from google import genai
        client = genai.Client()
        return lambda text: client.models.count_tokens(model=model_name, contents=text).total_tokens
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return lambda text: len(tokenizer.encode(text))

def regrade_file(filepath, apply=False, exp_name=None, model_name=None):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {filepath}: {e}")
            return False, 0, 0, 0, 0

    if isinstance(data, list):
        results = data
        is_list = True
    elif isinstance(data, dict) and 'results' in data:
        results = data['results']
        is_list = False
    else:
        print(f"Skipping {filepath}: Unrecognized format.")
        return False, 0, 0, 0, 0

    # Extract summary block if present
    summary_idx = -1
    for i, r in enumerate(results):
        if 'summary' in r:
            summary_idx = i
            break
            
    summary_block = None
    max_tokens = 32768
    if summary_idx != -1:
        summary_block = results.pop(summary_idx)
        max_tokens = summary_block['summary'].get('max_tokens', summary_block['summary'].get('max_model_length', 32768))
        if not model_name:
            model_name = summary_block['summary'].get('model', None)

    # If it's compound, we need model_name for exact token counting
    token_counter = None
    if exp_name == 'compound':
        if not model_name:
            parts = filepath.replace('\\', '/').split('/')
            try:
                results_idx = parts.index('results')
                model_name = parts[results_idx + 1]
            except (ValueError, IndexError):
                print(f"Warning: Could not infer model name for {filepath}")
                model_name = "unknown"
            
            if "_" in model_name and not any(x in model_name for x in ["gpt", "claude", "gemini"]):
                model_name = model_name.replace('_', '/', 1)
        
        try:
            token_counter = get_exact_token_counter(model_name)
        except Exception as e:
            print(f"Warning: Could not initialize tokenizer for {model_name} in {filepath}: {e}")

    changed_items = 0
    total = len(results)
    flips_correct = 0
    flips_incorrect = 0
    max_token_cutoffs = 0
    failures = 0

    print(f"\nProcessing: {filepath}")

    for r in results:
        output = r.get('output', '')
        gt = r.get('ground_truth', '')
        old_correct = r.get('correct', False)
        old_extracted = r.get('extracted', None)

        kwargs = {'exp_name': exp_name} if exp_name else {}
        new_extracted, new_correct = extract_and_grade(output, gt, **kwargs)

        if new_correct != old_correct or str(new_extracted) != str(old_extracted):
            changed_items += 1
            direction = ""
            if not old_correct and new_correct:
                direction = "  ✓ FIXED"
                flips_correct += 1
            elif old_correct and not new_correct:
                direction = "  ✗ REGRESSED"
                flips_incorrect += 1
            else:
                direction = "  ~ extracted changed"
                
            print(f"  ID {r.get('id', '?')}s{r.get('sample_idx', 0)}: "
                  f"\"{old_extracted}\" -> \"{new_extracted}\" "
                  f"({old_correct} -> {new_correct}){direction}")
                  
            r['extracted'] = str(new_extracted) if new_extracted is not None else None
            r['correct'] = new_correct

        # Precise token counting for compound mode
        if exp_name == 'compound' and token_counter:
            try:
                token_count = token_counter(output)
                r['output_tokens'] = token_count
                if token_count >= max_tokens * 0.98:
                    max_token_cutoffs += 1
            except Exception as e:
                pass # Just ignore if tokenizer fails

        if new_extracted is None or str(new_extracted).startswith("ERROR"):
            failures += 1

    # Format summary
    new_correct_count = sum(1 for r in results if r.get('correct', False))
    accuracy = new_correct_count / len(results) if len(results) > 0 else 0

    if summary_block:
        old_acc = summary_block['summary'].get('accuracy', 0)
        if abs(accuracy - old_acc) > 1e-6:
            changed_items += 1
            
        summary_block['summary']['accuracy'] = accuracy
        summary_block['summary']['correct'] = new_correct_count
        summary_block['summary']['total'] = total
        summary_block['summary']['failures'] = len(results) - new_correct_count
        if exp_name == 'compound' and token_counter:
            summary_block['summary']['max_token_cutoffs'] = max_token_cutoffs
        results.append(summary_block)

    if changed_items > 0:
        if apply:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Updated and saved to disk.")
        else:
            print(f"  Dry run. Detected {changed_items} changed items. Run with --apply to commit.")
    else:
        print("  Already up to date.")

    return changed_items > 0, total, changed_items, flips_correct, flips_incorrect

def main():
    parser = argparse.ArgumentParser(description='Unified regrading script.')
    parser.add_argument('--mode', choices=['aime2025', 'compound'], required=True, 
                        help='The grading strategy and experiment target to use.')
    parser.add_argument('--file', '-f', type=str, default=None, 
                        help='Specific JSON file to process.')
    parser.add_argument('--model', type=str, default=None,
                        help='Explicit model name. Pass "all" for bulk global processing logic.')
    parser.add_argument('--apply', action='store_true', 
                        help='Actually overwrite the JSON files in place.')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_process = []
    
    if args.file and args.model != 'all':
        files_to_process = [args.file]
    else:
        if args.mode == 'aime2025':
            pattern = os.path.join(base_dir, '*', 'results', '*', 'MathArena_aime_2025', '*.json')
            files_to_process = glob.glob(pattern)
        elif args.mode == 'compound':
            if args.model == 'all':
                pattern = os.path.join(base_dir, 'compound', 'results', '*', '*', '*.json')
                files_to_process = glob.glob(pattern)
            else:
                print("Error: For 'compound' mode, please specify a --file or use --model all to run globally.")
                return

    if not files_to_process:
        print("No result files to process matching criteria.")
        return
        
    print(f"Targeting {len(files_to_process)} result files...")
    
    exp_name = 'compound' if args.mode == 'compound' else None
    
    stale_files = 0
    total_items = 0
    total_changed = 0
    total_fixed = 0
    total_regressed = 0
    
    for f in files_to_process:
        if "raw" in os.path.basename(f).lower():
            continue # skip raw untokenized results
            
        m_name = args.model if args.model != 'all' else None
        
        changed, t, c, fixed, regressed = regrade_file(f, apply=args.apply, exp_name=exp_name, model_name=m_name)
        if changed:
            stale_files += 1
            total_changed += c
            total_fixed += fixed
            total_regressed += regressed
        total_items += t
            
    print(f"\n{'='*40}")
    print(f"Overall Summary ({'APPLY' if args.apply else 'DRY RUN'}):")
    print(f"Total files checked: {len(files_to_process)}")
    print(f"Files needing update:{stale_files}")
    print(f"Total items regraded:{total_items}")
    print(f"Total items changed: {total_changed}")
    print(f"Items Fixed (✓):     {total_fixed}")
    print(f"Items Regressed (✗): {total_regressed}")
    print(f"{'='*40}\n")
    if stale_files > 0 and not args.apply:
        print("Run again with --apply to commit these changes to disk.")

if __name__ == '__main__':
    main()
