#!/usr/bin/env python3
import os
import json
import glob
import sys

# Add experiments dir to path so we can import util
sys.path.insert(0, '/home/golikovp/Antigravity/Linguistic_traps/experiments')
from util import extract_and_grade

def regrade_file(filepath, apply=False):
    print(f"Processing: {filepath}")
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"  Error loading JSON: {e}")
            return False

    if not isinstance(data, list):
        print(f"  Skipping: Not a list format")
        return False

    correct_count = 0
    total_count = 0
    failures = 0
    clean_data = []
    summary_entry = None
    changed = False

    for entry in data:
        if 'summary' in entry:
            summary_entry = entry
            continue
        
        output = entry.get('output', '')
        gt = entry.get('ground_truth', '')
        old_correct = entry.get('correct', False)
        old_extracted = entry.get('extracted', None)
        
        extracted, is_correct = extract_and_grade(output, gt)
        
        if is_correct != old_correct or str(extracted) != str(old_extracted):
            changed = True
            entry['extracted'] = extracted
            entry['correct'] = is_correct
        
        total_count += 1
        if is_correct:
            correct_count += 1
        if extracted is None or (isinstance(extracted, str) and extracted.startswith("ERROR")):
            failures += 1
        
        clean_data.append(entry)

    if total_count > 0:
        accuracy = correct_count / total_count
        if summary_entry:
            old_acc = summary_entry['summary'].get('accuracy', 0)
            if abs(accuracy - old_acc) > 1e-6:
                changed = True
                print(f"  Accuracy change: {old_acc:.2%} -> {accuracy:.2%}")
            
            summary_entry['summary']['accuracy'] = accuracy
            summary_entry['summary']['correct'] = correct_count
            summary_entry['summary']['total'] = total_count
            summary_entry['summary']['failures'] = failures
            clean_data.append(summary_entry)
        
        if changed and apply:
            with open(filepath, 'w') as f:
                json.dump(clean_data, f, indent=2)
            print(f"  Updated in place.")
        elif changed:
            print(f"  Stale results detected (Dry Run).")
        else:
            print(f"  Already up to date.")
    
    return changed

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regrade all AIME 2025 results")
    parser.add_argument("--apply", action="store_true", help="Apply changes in place")
    args = parser.parse_args()

    base_dir = '/home/golikovp/Antigravity/Linguistic_traps/experiments'
    # Pattern to find AIME 2025 results across all techniques and models
    pattern = os.path.join(base_dir, '*', 'results', '*', 'MathArena_aime_2025', '*.json')
    files = glob.glob(pattern)

    if not files:
        print("No AIME 2025 result files found.")
        return

    print(f"Found {len(files)} result files.")
    
    stale_count = 0
    for f in files:
        if regrade_file(f, apply=args.apply):
            stale_count += 1
            
    print(f"\nSummary:")
    print(f"Total files: {len(files)}")
    print(f"Stale files: {stale_count}")
    if not args.apply and stale_count > 0:
        print("\nRun with --apply to commit changes.")

if __name__ == "__main__":
    main()
