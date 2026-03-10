#!/usr/bin/env python3
"""
Re-grade Gemini result JSON files using the updated extract_and_grade function.

Walks through Gemini result files under experiments/, re-extracts answers (now
with bold-notation fallback), re-grades them, and updates the files in place.
Only processes files whose path contains 'gemini' or whose metadata model
field contains 'gemini'. Prints a summary of changes.

Usage:
    python regrade_results.py                          # dry run (default)
    python regrade_results.py --apply                  # overwrite files
    python regrade_results.py --path context_saturation/results/gemini-3.1-pro-preview  # specific dir
"""

import os
import sys
import json
import glob
import argparse

# Add experiments dir to path so we can import util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import extract_and_grade


def regrade_file(filepath, dry_run=True):
    """Re-grade a single results JSON file. Returns (changed_count, total_count, flips)."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle both formats: list of results, or dict with 'results' key
    if isinstance(data, list):
        results = data
        is_list = True
    elif isinstance(data, dict) and 'results' in data:
        results = data['results']
        is_list = False
    else:
        return 0, 0, []

    changed = 0
    flips = []
    for r in results:
        output = r.get('output', '')
        gt = r.get('ground_truth', '')
        old_correct = r.get('correct', False)
        old_extracted = r.get('extracted', None)

        new_extracted, new_correct = extract_and_grade(output, gt)

        if new_correct != old_correct or str(new_extracted) != str(old_extracted):
            changed += 1
            flips.append({
                'id': r.get('id', '?'),
                'sample_idx': r.get('sample_idx', 0),
                'old_extracted': old_extracted,
                'new_extracted': new_extracted,
                'old_correct': old_correct,
                'new_correct': new_correct,
            })
            r['extracted'] = str(new_extracted) if new_extracted is not None else None
            r['correct'] = new_correct

    # Update statistics if present
    if not is_list and 'statistics' in data:
        new_correct_count = sum(1 for r in results if r.get('correct', False))
        new_failures = len(results) - new_correct_count
        data['statistics']['correct'] = new_correct_count
        data['statistics']['failures'] = new_failures

    if changed > 0 and not dry_run:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return changed, len(results), flips


def is_gemini_file(filepath):
    """Check if a result file is from a Gemini model (by path or metadata)."""
    # Quick path-based check first
    if 'gemini' in filepath.lower():
        return True
    # Fall back to reading metadata
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            model = data.get('metadata', {}).get('model', '')
            return 'gemini' in model.lower()
        elif isinstance(data, list) and data:
            model = data[0].get('model', '')
            return 'gemini' in model.lower()
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description='Re-grade Gemini result files with updated extract_and_grade')
    parser.add_argument('--apply', action='store_true', help='Actually overwrite files (default: dry run)')
    parser.add_argument('--path', type=str, default=None,
                        help='Specific directory or file to re-grade (relative to experiments/)')
    args = parser.parse_args()

    experiments_dir = os.path.dirname(os.path.abspath(__file__))

    if args.path:
        search_root = os.path.join(experiments_dir, args.path)
    else:
        search_root = experiments_dir

    # Find all result JSON files
    if os.path.isfile(search_root):
        json_files = [search_root]
    else:
        json_files = sorted(glob.glob(os.path.join(search_root, '**', '*.json'), recursive=True))

    # Filter to only result files (heuristic: filename contains 'result' or is in a 'results' dir)
    result_files = [
        f for f in json_files
        if 'result' in os.path.basename(f).lower() or '/results/' in f
    ]
    # Exclude tracking, cache, batch metadata files
    result_files = [
        f for f in result_files
        if not any(x in os.path.basename(f).lower() for x in ['tracking', 'cache', 'batch_tracking', 'jobs_'])
    ]

    # Filter to Gemini-only files
    result_files = [f for f in result_files if is_gemini_file(f)]

    if not result_files:
        print(f"No result files found under {search_root}")
        return

    mode = "DRY RUN" if not args.apply else "APPLYING CHANGES"
    print(f"\n{'='*70}")
    print(f"  RE-GRADING RESULTS ({mode})")
    print(f"  Scanning: {search_root}")
    print(f"  Found: {len(result_files)} result file(s)")
    print(f"{'='*70}\n")

    total_changed = 0
    total_items = 0
    total_flips_correct = 0
    total_flips_incorrect = 0

    for filepath in result_files:
        rel = os.path.relpath(filepath, experiments_dir)
        changed, total, flips = regrade_file(filepath, dry_run=not args.apply)
        total_items += total

        if changed > 0:
            total_changed += changed
            print(f"\n  {rel}  ({changed}/{total} changed)")
            for flip in flips:
                direction = ""
                if not flip['old_correct'] and flip['new_correct']:
                    direction = "  ✓ FIXED"
                    total_flips_correct += 1
                elif flip['old_correct'] and not flip['new_correct']:
                    direction = "  ✗ REGRESSED"
                    total_flips_incorrect += 1
                else:
                    direction = "  ~ extracted changed"
                print(f"    ID {flip['id']}s{flip['sample_idx']}: "
                      f"\"{flip['old_extracted']}\" → \"{flip['new_extracted']}\" "
                      f"({flip['old_correct']} → {flip['new_correct']}){direction}")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"  Files scanned:    {len(result_files)}")
    print(f"  Items re-graded:  {total_items}")
    print(f"  Items changed:    {total_changed}")
    print(f"  Newly correct:    {total_flips_correct}")
    print(f"  Regressions:      {total_flips_incorrect}")
    if not args.apply and total_changed > 0:
        print(f"\n  Run with --apply to save changes")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
