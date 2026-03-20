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
    if summary_idx != -1:
        # separate it
        summary_block = results.pop(summary_idx)

    changed = 0
    total = len(results)
    flips_correct = 0
    flips_incorrect = 0

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

    # Restore summary block with updated stats if it existed
    if summary_block:
        new_correct_count = sum(1 for r in results if r.get('correct', False))
        new_failures = len(results) - new_correct_count
        summary_block['summary']['correct'] = new_correct_count
        summary_block['summary']['failures'] = new_failures
        accuracy = new_correct_count / len(results) if len(results) > 0 else 0
        summary_block['summary']['accuracy'] = accuracy
        results.append(summary_block)
        
    print(f"\n--- Summary ---")
    print(f"Items regraded: {total}")
    print(f"Changed items:  {changed}")
    print(f"Fixed:          {flips_correct}")
    print(f"Regressed:      {flips_incorrect}")

    if summary_block:
        print(f"New Accuracy:   {accuracy:.2%} ({new_correct_count}/{total})")

    if args.apply and changed > 0:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Updated file saved.")
    elif changed > 0:
        print("Run with --apply to save changes.")

if __name__ == '__main__':
    main()
