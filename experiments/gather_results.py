
import argparse
import os
import sys
import json
import glob
from prettytable import PrettyTable

# Add local directory to path to import util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util import extract_and_grade

def get_latest_result_file(base_dir, exp_name, model_name, dataset_name, results_subdir="results"):
    # Sanitize names to match evaluate.py logic
    safe_model = model_name.replace('/', '_').replace(' ', '_')
    safe_dataset = dataset_name.replace('/', '_')
    
    # Path construction: experiments/<exp_name>/<results_subdir>/<safe_model>/<safe_dataset>/
    target_dir = os.path.join(base_dir, exp_name, results_subdir, safe_model, safe_dataset)
    
    if not os.path.exists(target_dir):
        return None, f"Directory not found: {target_dir}"
        
    # Find all json files
    pattern = os.path.join(target_dir, "*.json")
    files = glob.glob(pattern)
    
    # Filter out "raw" files? evaluate.py deletes them, but just in case
    files = [f for f in files if not f.endswith("_raw.json")]
    
    if not files:
        return None, f"No result files found in {target_dir}"
        
    # Sort by modification time (or filename timestamp)
    # Using mtime is generally reliable for "latest"
    latest_file = max(files, key=os.path.getmtime)
    return latest_file, None

def process_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return None, f"Error reading {filepath}: {e}"
        
    # Handle list or dict format
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict) and 'results' in data:
        results = data['results']
    else:
        return None, "Unknown JSON format"
        
    stats = {"correct": 0, "total": 0, "failures": 0}
    
    for res in results:
        # Re-extract and Re-verify using Math-Verify
        output = res.get('output', '')
        ground_truth = res.get('ground_truth')
        
        extracted, is_correct = extract_and_grade(output, ground_truth)
        
        stats['total'] += 1
        if is_correct:
            stats['correct'] += 1
        
        # Failure tracking
        if not extracted or (isinstance(extracted, str) and extracted.startswith('ERROR')):
            stats['failures'] += 1
            
    return stats, None

def main():
    parser = argparse.ArgumentParser(description="Gather and Re-score Results")
    parser.add_argument("--transformations", type=str, required=True, help="Comma-separated list of transformation names")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="Dataset name")
    parser.add_argument("--results_subdir", type=str, default="results", help="Subdirectory for results (e.g. results, results_agent, results_context)")
    
    args = parser.parse_args()
    
    transformations = [t.strip() for t in args.transformations.split(',') if t.strip()]
    if args.transformations == "all":
        transformations = ["baseline", "interleaved_context_line", "interleaved_context_word", 
        "interleaved_context_symbol", "not_not", "opposites", "rail_fence", "sentence_reversal", "split_reversal", 
        "word_reversal", "wrappers"]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    table = PrettyTable()
    table.field_names = ["Transformation", "Model", "Dataset", "Accuracy", "Correct/Total", "Status"]
    table.align = "l"
    
    print(f"Gathering results for model='{args.model}' on dataset='{args.dataset}' from '{args.results_subdir}'...")
    
    results_summary = []
    
    for trans in transformations:
        filepath, error = get_latest_result_file(base_dir, trans, args.model, args.dataset, args.results_subdir)
        
        if error:
            table.add_row([trans, args.model, args.dataset, "N/A", "0/0", "Missing/Error"])
            print(f"[{trans}] {error}")
            continue
            
        print(f"[{trans}] Processing {os.path.basename(filepath)}...")
        stats, err = process_file(filepath)
        
        if err:
            table.add_row([trans, args.model, args.dataset, "N/A", "0/0", f"Error: {err}"])
            continue
            
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        acc_str = f"{acc:.2%}"
        count_str = f"{stats['correct']}/{stats['total']}"
        
        table.add_row([trans, args.model, args.dataset, acc_str, count_str, "OK"])
        
        results_summary.append({
            "Transformation": trans,
            "Accuracy": acc_str,
            "Count": count_str
        })
        
    print("\n" + str(table))

if __name__ == "__main__":
    main()
