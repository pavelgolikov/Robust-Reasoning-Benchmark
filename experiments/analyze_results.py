import json
import os
import re
import glob
import argparse
import time

def find_latest_result(base_dir, technique, model_name, dataset_name):
    # Construct path: experiments/{technique}/results/{model}/{dataset}/*.json
    # safe names logic
    safe_model = model_name.replace('/', '_').replace(' ', '_')
    safe_dataset = dataset_name.replace('/', '_')
    
    search_dir = os.path.join(base_dir, technique, "results", safe_model, safe_dataset)
    if not os.path.exists(search_dir):
        # Fallback to old structure? No, user explicitly migrated.
        return None
        
    files = glob.glob(os.path.join(search_dir, "*.json"))
    # Filter out semantic/raw if any (though raw are deleted usually)
    files = [f for f in files if "semantic" not in f and "raw" not in f]
    
    if not files:
        return None
        
    return max(files, key=os.path.getmtime)

def analyze_single_file(result_file, print_details=True):
    if not os.path.exists(result_file):
        if print_details: print(f"File not found: {result_file}")
        return None

    with open(result_file, 'r') as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    extraction_failures = 0
    max_len_found = 0
    truncated_suspects = 0

    if print_details:
        print(f"Analyzing {total} samples from {result_file}...")

    for entry in data:
        output = entry.get('output', '')
        extracted = entry.get('extracted')
        is_correct = entry.get('correct', False)
        length = len(output)
        max_len_found = max(max_len_found, length)

        if is_correct:
            correct += 1
        
        if extracted is None or (isinstance(extracted, str) and extracted.startswith("ERROR")):
            extraction_failures += 1
            # Check for truncation in failures
            # Unclosed \boxed
            boxed_matches = list(re.finditer(r"\\boxed\{", output))
            if boxed_matches:
                last_boxed = boxed_matches[-1]
                if "}" not in output[last_boxed.end():]:
                    truncated_suspects += 1
            elif len(output) > 0 and output.strip()[-1] not in ['.', '!', '?', '}', '>', ']']:
                truncated_suspects += 1
        
    # Pass@1 equivalent (aggregated by ID)
    by_id = {}
    for entry in data:
        eid = entry.get('id')
        if eid not in by_id: by_id[eid] = []
        by_id[eid].append(entry.get('correct'))
    
    pass_rates = [sum(v)/len(v) for v in by_id.values()]
    avg_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct/total if total > 0 else 0,
        "failures": extraction_failures,
        "avg_pass_rate": avg_pass_rate,
        "file": os.path.basename(result_file)
    }

    if print_details:
        print("\nSummary:")
        print(f"Total Samples: {total}")
        print(f"Accuracy: {summary['accuracy']:.2%} ({correct}/{total})")
        print(f"Max Output Length: {max_len_found} chars")
        print(f"Extraction Failures: {extraction_failures}")
        print(f"Truncation Suspects (in Failures): {truncated_suspects}")
        print(f"Problem-Level Average Pass Rate: {avg_pass_rate:.2%}")
    
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to results JSON file (Single file mode)")
    parser.add_argument("--model", type=str, help="Model name (Batch mode)")
    parser.add_argument("--dataset", type=str, help="Dataset name (Batch mode)")
    parser.add_argument("--names", type=str, default="all", help="List of experiment names for batch mode")
    args = parser.parse_args()

    # Single File Mode
    if args.file:
        analyze_single_file(args.file)
        return

    # Batch Mode
    if args.model and args.dataset:
        if args.names == 'all':
            experiment_names = [ 'context_saturation', 'interleaved_context_line', 'interleaved_context_word',
            'not_not_yot', 'opposites', 'sentence_reversal', 'word_reversal', 'word_split_swap', 'wrappers', 'baseline' ] # Added baseline
        else:
            experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]
        
        # Sort to keep consistent order (baseline first if present or alphabetical)
        experiment_names.sort()
        if 'baseline' in experiment_names:
            experiment_names.remove('baseline')
            experiment_names.insert(0, 'baseline')

        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        results_rows = []
        
        print(f"Batch Analysis for Model: {args.model}, Dataset: {args.dataset}")
        
        for name in experiment_names:
            latest_file = find_latest_result(base_dir, name, args.model, args.dataset)
            if latest_file:
                stats = analyze_single_file(latest_file, print_details=False)
                if stats:
                    stats['name'] = name
                    results_rows.append(stats)
            else:
                # print(f"  No results found for {name}")
                pass

        # Print Table
        print(f"\nResults for {args.model} on {args.dataset}")
        header = f"{'Experiment':<30} | {'Total':<8} | {'Accuracy':<10} | {'Failures':<10} | {'File'}"
        print(header)
        print("-" * len(header))
        
        for row in results_rows:
            print(f"{row['name']:<30} | {row['total']:<8} | {row['accuracy']:<10.2%} | {row['failures']:<10} | {row['file']}")
        return

    print("Please provide either --file OR (--model, --dataset, --names)")

if __name__ == "__main__":
    main()
