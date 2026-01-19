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

def normalize_text(text, method='standard'):
    if method == 'aggressive':
        # Remove all whitespace for very robust comparison (e.g. context_saturation LaTeX issues)
        return "".join(text.split())
    # Always normalize whitespace for robust comparison as requested
    return " ".join(text.split())

def main():
    parser = argparse.ArgumentParser(description="Test Reversibility of Transformations")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of techniques")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of examples to test per technique")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="Dataset to test on")
    parser.add_argument("--output", type=str, help="Output report file (default: results/<dataset>_reversibility_report.txt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_distractors", type=int, default=2, help="Number of distractors for context_saturation")
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
    
    if 'aime' in args.dataset:
        split = 'train'
    else:
        split = 'all'
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    try:
        dataset = load_dataset(args.dataset, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.names == 'all':
        experiment_names = ['context_saturation',
                            'interleaved_context_line',
                            'interleaved_context_word',
                            'not_not_yot',
                            'opposites',
                            'sentence_reversal',
                            'word_reversal',
                            'word_split_swap',
                            'wrappers']
    else:
        experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]
    
    report_lines = []
    report_lines.append(f"Reversibility Test Report - {args.num_samples} samples per technique - Dataset: {args.dataset}")
    report_lines.append("="*80)
    
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

        # Select random samples
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
        
        passed = 0
        total = 0
        
        for i in indices:
            total += 1
            original_raw = dataset[i]['problem']
            
            # Pre-processing: Remove empty lines (lines with only whitespace)
            # This ensures that context_saturation (which splits on \n\n) doesn't fragment the real problem.
            original_raw = "\n".join([line for line in original_raw.splitlines() if line.strip()])

            
            # Prepare args for apply (some need context)
            kwargs = {}
            if exp_name in ['interleaved_context_line', 'interleaved_context_word', 'interleaved_substitutions']:
                 next_idx = (i + 1) % len(dataset)
                 problem_b = dataset[next_idx]['problem']
                 kwargs = {'problem_b': problem_b}
            elif exp_name == 'context_saturation':
                 kwargs = {'num_distractors': args.num_distractors}
                 
            try:
                if 'problem_b' in kwargs:
                     transformed = apply_func(original_raw, kwargs['problem_b'], seed=args.seed)
                elif exp_name == 'context_saturation':
                     transformed = apply_func(original_raw, kwargs['num_distractors'], seed=args.seed)
                elif exp_name == 'opposites':
                     # Force 100% swap for reversibility testing to ensure no partial swaps exist
                     transformed = apply_func(original_raw, k=1, seed=args.seed)
                else:
                    transformed = apply_func(original_raw, seed=args.seed)
                    
                reversed_text = reverse_func(transformed)
                
                # Normalize for comparison
                norm_method = 'standard'
                if exp_name == 'context_saturation' or exp_name == 'opposites':
                    norm_method = 'aggressive'
                elif exp_name in ['interleaved_context_line', 'interleaved_context_word']:
                    norm_method = 'interleaved_context'
                    
                norm_orig = normalize_text(original_raw, norm_method)
                norm_rev = normalize_text(reversed_text, norm_method)
                
                is_match = (norm_orig == norm_rev)
                if exp_name in ['interleaved_context_line', 'interleaved_context_word']:
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
                    
                report_lines.append(f"\n{'-'*80}")
                report_lines.append(f"Sample ID: {i} | Status: {status}")
                report_lines.append(f"{'-'*80}")
                
                report_lines.append("\n[TRANSFORMED PROBLEM]:")
                report_lines.append(transformed)
                
                report_lines.append("\n[ORIGINAL PROBLEM]:")
                report_lines.append(original_raw)
                
                report_lines.append("\n[REVERSED PROBLEM]:")
                report_lines.append(reversed_text)
                
                if not is_match:
                     report_lines.append("\n[COMPARISON - NORMALIZED]:")
                     report_lines.append("--- Original (Norm) ---")
                     report_lines.append(norm_orig)
                     report_lines.append("--- Reversed (Norm) ---")
                     report_lines.append(norm_rev)
                     report_lines.append(f"--- Length Diff: {len(norm_rev) - len(norm_orig)} ---")
                     
            except Exception as e:
                report_lines.append(f"\nSample ID: {i} | ERROR during transform/reverse: {e}")

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
