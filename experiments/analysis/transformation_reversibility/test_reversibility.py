import argparse
import os
import sys
import random
import importlib
from datasets import load_dataset

# Add project root (parent of experiments dir) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir: experiments/analysis/transformation_reversibility
# dirname: experiments/analysis
# dirname: experiments
# dirname: project_root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

if project_root not in sys.path:
    sys.path.append(project_root)

# Also add experiments dir for util.py internal imports
experiments_dir = os.path.join(project_root, 'experiments')
if experiments_dir not in sys.path:
    sys.path.append(experiments_dir)

from experiments.util import remove_latex_comments, sanitize_inverted_escapes, flatten_text

def normalize_text(text, method='standard'):
    if method == 'aggressive':
        # Remove all whitespace for very robust comparison (e.g. context_saturation LaTeX issues)
        return "".join(text.split())
    # Always normalize whitespace for robust comparison as requested
    return " ".join(text.split())

def main():
    parser = argparse.ArgumentParser(description="Test Reversibility of Transformations")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of techniques")
    parser.add_argument("--limit", type=str, default="0", help="Number of examples (N) or range (start:end) to test. 0 means all.")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="Dataset to test on")
    parser.add_argument("--output", type=str, help="Output report file (default: results/<dataset>_reversibility_report.txt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_distractors", type=int, default=2, help="Number of distractors for context_saturation")
    parser.add_argument("--split", type=str, default="all", help="Split to use for testing")
    parser.add_argument("--num_print_samples", type=int, default=1, help="Number of matching samples to print in the report.")
    args = parser.parse_args()

    # helper for safe name
    safe_dataset_name = args.dataset.replace('/', '_')
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: results/<dataset>_reversibility_report.txt inside script dir
        results_dir = os.path.join(script_dir, "results")
        output_path = os.path.join(results_dir, f"{safe_dataset_name}_reversibility_report.txt")

    random.seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    try:
        dataset = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.names == 'all':
        # experiment_names = ['context_saturation',
        experiment_names = ['interleaved_context_line',
                            'interleaved_context_word',
                            'interleaved_context_symbol',
                            'not_not',
                            'opposites',
                            'sentence_reversal',
                            'word_reversal',
                            # 'word_split_swap',
                            'wrappers',
                            'split_reversal',
                            'rail_fence']
    else:
        experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]
    
    sample_info = f"samples: {args.limit}"
    report_lines = []
    report_lines.append(f"Reversibility Test Report - {sample_info} - Dataset: {args.dataset}")
    report_lines.append("="*80)
    
    # Parse limit/range once
    limit_str = str(args.limit).strip().lower()
    indices = []
    
    if limit_str in ['0', 'all', 'none']:
        indices = list(range(len(dataset)))
    elif ':' in limit_str:
        # Range: start:end
        parts = limit_str.split(':')
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else len(dataset)
        # Ensure bounds
        start = max(0, start)
        end = min(end, len(dataset))
        indices = list(range(start, end))
    else:
        # First N
        try:
            count = int(limit_str)
            if count == 0:
                 indices = list(range(len(dataset)))
            else:
                 indices = list(range(min(count, len(dataset))))
        except ValueError:
            print(f"Error: Invalid limit format '{limit_str}'. Use integer or start:end.")
            return

    print(f"Selected {len(indices)} samples (Limit: {args.limit})")

    for exp_name in experiment_names:
        print(f"Testing {exp_name}...")
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"EXPERIMENT: {exp_name}")
        report_lines.append(f"{'='*40}")
        
        # Dynamic import
        try:
            module_name = f"experiments.{exp_name}.transformation"
            module = importlib.import_module(module_name)
            
            apply_func_name = f"apply_{exp_name}"
            reverse_func_name = f"reverse_{exp_name}"
            
            if not hasattr(module, apply_func_name):
                report_lines.append(f"ERROR: Function {apply_func_name} not found in {module_name}")
                continue
                
            apply_func = getattr(module, apply_func_name)
            
            if not hasattr(module, reverse_func_name):
                report_lines.append(f"WARNING: Function {reverse_func_name} not found. Skipping reversibility check.")
                continue
                
            reverse_func = getattr(module, reverse_func_name)
            
        except ImportError as e:
            report_lines.append(f"ERROR: Could not import module for {exp_name}: {e}")
            continue
        
        passed = 0
        total = 0
        saved_matches = []
        
        for i in indices:
            total += 1
            original_raw = dataset[i]['problem']
            
            # Pre-processing: Remove empty lines (lines with only whitespace)
            # This ensures that context_saturation (which splits on \n\n) doesn't fragment the real problem.
            original_raw = "\n".join([line for line in original_raw.splitlines() if line.strip()])

            
            # Prepare args for apply (some need context)
            kwargs = {}
            if exp_name in ['interleaved_context_line', 'interleaved_context_word', 'interleaved_context_symbol', 'interleaved_substitutions']:
                 next_idx = (i + 1) % len(dataset)
                 problem_b = remove_latex_comments(dataset[next_idx]['problem'])
                 problem_b = sanitize_inverted_escapes(problem_b)
                 problem_b = flatten_text(problem_b)
                 kwargs = {'problem_b': problem_b}
            elif exp_name == 'context_saturation':
                 kwargs = {'num_distractors': args.num_distractors}
            elif exp_name == 'rail_fence':
                 kwargs = {'num_rails': 3}
                 
            try:
                # 0. Global Sanitization
                original_raw = remove_latex_comments(original_raw)
                original_raw = sanitize_inverted_escapes(original_raw)
                original_raw = flatten_text(original_raw)

                if 'problem_b' in kwargs:
                     transformed = apply_func(original_raw, kwargs['problem_b'], seed=args.seed)
                elif exp_name == 'context_saturation':
                     transformed = apply_func(original_raw, kwargs['num_distractors'], seed=args.seed)
                elif exp_name == 'opposites':
                     # Force 100% swap for reversibility testing to ensure no partial swaps exist
                     transformed = apply_func(original_raw, k=1, seed=args.seed)
                elif exp_name == 'rail_fence':
                     transformed = apply_func(original_raw, kwargs['num_rails'])
                else:
                    transformed = apply_func(original_raw, seed=args.seed)
                    
                reversed_text = reverse_func(transformed)
                
                # Normalize for comparison
                norm_method = 'standard'
                if exp_name == 'context_saturation' or exp_name == 'opposites':
                    norm_method = 'aggressive'
                elif exp_name in ['interleaved_context_line', 'interleaved_context_word', 'interleaved_context_symbol']:
                    norm_method = 'interleaved_context'
                    
                norm_orig = normalize_text(original_raw, norm_method)
                norm_rev = normalize_text(reversed_text, norm_method)
                
                is_match = (norm_orig == norm_rev)
                if exp_name in ['interleaved_context_line', 'interleaved_context_word', 'interleaved_context_symbol']:
                    # interleaved_context cycles content, so reversed output may contain repetitions.
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
                    # Collect matches up to limit
                    if len(saved_matches) < args.num_print_samples:
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

        # Report collected matches
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
        report_lines.append(f"{'='*40}\n")
    
    # Ensure output directory exists (experiments/ usually exists)
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Write output
    with open(output_path, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"Report written to {output_path}")

if __name__ == "__main__":
    main()
