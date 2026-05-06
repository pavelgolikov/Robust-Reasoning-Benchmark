#!/usr/bin/env python3
"""
Standalone Transformation Reversibility Demo
=============================================
Downloads AIME 2024 and AIME 2025 datasets, applies every transformation
to every question, and verifies reversibility. Prints sample transformed
problems for review.

Usage:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python demo.py

Output is written to demo_output/ directory.
"""

import argparse
import os
import sys
import re
import random
import importlib

# ── Ensure transformations/ is importable ────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# ── NLTK / spaCy bootstrap ──────────────────────────────────────────
import nltk
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception:
    pass

# ── Utility functions (inlined from experiments/util.py) ─────────────

def remove_latex_comments(text):
    """Removes LaTeX comments (starting with %, unless escaped as \\%)."""
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        def replacer(match):
            if match.group(1):
                return match.group(1)
            else:
                return ""
        clean_line = re.sub(r'(\\\\|\\%)|(%.*)', replacer, line)
        clean_lines.append(clean_line)
    return "\n".join(clean_lines)


def sanitize_inverted_escapes(text):
    """Inserts a space between specific characters and a backslash to prevent
    accidental escape formation when text is reversed."""
    return re.sub(r'([bntafr])\\', r'\1 \\', text)


def flatten_text(text):
    """Standardizes text by replacing newlines with '; '."""
    if text is None:
        return ""
    return text.replace('\n', '; ')


def normalize_text(text):
    """Normalize whitespace for robust comparison."""
    return " ".join(text.split())


# ── Configuration ────────────────────────────────────────────────────

DATASETS = [
    "HuggingFaceH4/aime_2024",
    "MathArena/aime_2025",
]

EXPERIMENT_NAMES = [
    'interleaved_context_line',
    'interleaved_context_word',
    'interleaved_context_symbol',
    'not_not',
    'opposites',
    'rail_fence',
    'sentence_reversal',
    'split_reversal',
    'word_reversal',
    'wrappers',
    'snake_horizontal',
    'snake_vertical',
    'rectangle_perimeter',
]

NUM_PRINT_SAMPLES = 30
SEED = 42


# ── Core logic ───────────────────────────────────────────────────────

def load_transformation(exp_name):
    """Dynamically import apply_ and reverse_ functions from transformations/<exp_name>/transformation.py."""
    module_name = f"transformations.{exp_name}.transformation"
    module = importlib.import_module(module_name)

    apply_func_name = f"apply_{exp_name}"
    reverse_func_name = f"reverse_{exp_name}"

    apply_func = getattr(module, apply_func_name, None)
    reverse_func = getattr(module, reverse_func_name, None)

    return apply_func, reverse_func


def run_demo(dataset_name, dataset, report_lines, num_print_samples=NUM_PRINT_SAMPLES, seed=SEED):
    """Run all transformations on a single dataset and append results to report_lines."""
    report_lines.append(f"\n{'#'*80}")
    report_lines.append(f"# DATASET: {dataset_name}")
    report_lines.append(f"# Samples: {len(dataset)}")
    report_lines.append(f"{'#'*80}\n")

    random.seed(seed)
    indices = list(range(len(dataset)))

    for exp_name in EXPERIMENT_NAMES:
        print(f"  Testing {exp_name}...")
        report_lines.append(f"\n{'='*60}")
        report_lines.append(f"EXPERIMENT: {exp_name}")
        report_lines.append(f"{'='*60}")

        try:
            apply_func, reverse_func = load_transformation(exp_name)
        except ImportError as e:
            report_lines.append(f"ERROR: Could not import module for {exp_name}: {e}")
            continue

        if apply_func is None:
            report_lines.append(f"ERROR: apply_{exp_name} not found.")
            continue
        if reverse_func is None:
            report_lines.append(f"WARNING: reverse_{exp_name} not found. Skipping reversibility check.")
            continue

        passed = 0
        total = 0
        saved_matches = []

        for i in indices:
            total += 1
            original_raw = dataset[i]['problem']
            # Remove empty lines
            original_raw = "\n".join([line for line in original_raw.splitlines() if line.strip()])

            # Prepare kwargs
            kwargs = {}
            if exp_name in ['interleaved_context_line', 'interleaved_context_word', 'interleaved_context_symbol']:
                next_idx = (i + 1) % len(dataset)
                problem_b = remove_latex_comments(dataset[next_idx]['problem'])
                problem_b = sanitize_inverted_escapes(problem_b)
                problem_b = flatten_text(problem_b)
                kwargs = {'problem_b': problem_b}
            elif exp_name == 'rail_fence':
                kwargs = {'num_rails': 3}

            try:
                # Global sanitization
                original_raw = remove_latex_comments(original_raw)
                original_raw = sanitize_inverted_escapes(original_raw)
                original_raw = flatten_text(original_raw)

                # Apply transformation
                if 'problem_b' in kwargs:
                    transformed = apply_func(original_raw, kwargs['problem_b'], seed=seed)
                elif exp_name == 'opposites':
                    transformed = apply_func(original_raw, k=1, seed=seed)
                elif exp_name == 'rail_fence':
                    transformed = apply_func(original_raw, kwargs['num_rails'])
                elif exp_name in ['rectangle_perimeter', 'snake_vertical', 'snake_horizontal']:
                    transformed = apply_func(original_raw)
                else:
                    transformed = apply_func(original_raw, seed=seed)

                # Reverse transformation
                reversed_text = reverse_func(transformed)

                # Normalize for comparison
                norm_orig = normalize_text(original_raw)
                norm_rev = normalize_text(reversed_text)

                is_match = (norm_orig == norm_rev)
                if exp_name in ['interleaved_context_line', 'interleaved_context_word', 'interleaved_context_symbol']:
                    if norm_rev.startswith(norm_orig):
                        is_match = True
                        status = "MATCH (Prefix/Cycled)"
                    else:
                        is_match = False
                        status = "MISMATCH"
                else:
                    status = "MATCH" if is_match else "MISMATCH"

                if is_match:
                    passed += 1
                    if len(saved_matches) < num_print_samples:
                        saved_matches.append((i, status, transformed, original_raw, reversed_text))

                if not is_match:
                    report_lines.append(f"\n{'-'*80}")
                    report_lines.append(f"Sample ID: {i} | Status: {status}")
                    report_lines.append(f"{'-'*80}")
                    report_lines.append("\n[TRANSFORMED PROBLEM]:")
                    report_lines.append(transformed)
                    report_lines.append("\n[ORIGINAL PROBLEM]:")
                    report_lines.append(original_raw)
                    report_lines.append("\n[REVERSED PROBLEM]:")
                    report_lines.append(reversed_text)
                    report_lines.append("\n[COMPARISON - NORMALIZED]:")
                    report_lines.append("--- Original (Norm) ---")
                    report_lines.append(norm_orig)
                    report_lines.append("--- Reversed (Norm) ---")
                    report_lines.append(norm_rev)
                    report_lines.append(f"--- Length Diff: {len(norm_rev) - len(norm_orig)} ---")

            except Exception as e:
                report_lines.append(f"\nSample ID: {i} | ERROR during transform/reverse: {e}")

        # Report collected match examples
        for match_details in saved_matches:
            idx, st, tr, orig, rev = match_details
            report_lines.append(f"\n{'-'*40}")
            report_lines.append(f"MATCH EXAMPLE (Sample ID: {idx})")
            report_lines.append(f"{'-'*40}")
            report_lines.append("\n[TRANSFORMED PROBLEM]:")
            report_lines.append(tr)
            report_lines.append("\n[ORIGINAL PROBLEM]:")
            report_lines.append(orig)
            report_lines.append("\n[REVERSED PROBLEM]:")
            report_lines.append(rev)

        report_lines.append(f"\n{'-'*40}")
        report_lines.append(f"Result: {passed}/{total} Passed")
        report_lines.append(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone demo: apply and reverse all transformations on AIME 2024 & 2025."
    )
    parser.add_argument("--num_print_samples", type=int, default=NUM_PRINT_SAMPLES,
                        help="Number of matching samples to print per transformation (default: 30)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default="demo_output",
                        help="Directory to write reports into (default: demo_output)")
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)

    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    from datasets import load_dataset

    for dataset_name in DATASETS:
        safe_name = dataset_name.replace('/', '_')
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name}...")
        print(f"{'='*60}")

        try:
            dataset = load_dataset(dataset_name, split="all")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        print(f"Loaded {len(dataset)} samples.")

        report_lines = []
        report_lines.append(f"Transformation Reversibility Demo Report")
        report_lines.append(f"Dataset: {dataset_name} ({len(dataset)} samples)")
        report_lines.append(f"Seed: {args.seed}")
        report_lines.append(f"Print samples per transformation: {args.num_print_samples}")
        report_lines.append(f"{'='*80}\n")

        run_demo(dataset_name, dataset, report_lines, num_print_samples=args.num_print_samples, seed=seed)

        output_path = os.path.join(output_dir, f"{safe_name}_reversibility_report.txt")
        with open(output_path, 'w') as f:
            f.write("\n".join(report_lines))
        print(f"\nReport written to: {output_path}")

    print("\n" + "="*60)
    print("Demo complete! Reports are in:", output_dir)
    print("="*60)


if __name__ == "__main__":
    main()
