import argparse
import os
import json
import glob
import re
from datetime import datetime

def get_latest_result_file(exp_dir, safe_model, safe_dataset):
    """Finds the latest JSON result file for a given experiment, model, and dataset."""
    search_path = os.path.join(exp_dir, "results", safe_model, safe_dataset, "*.json")
    files = glob.glob(search_path)
    
    # Filter out _raw.json and _semantic_analysis.json
    valid_files = [f for f in files if not f.endswith("_raw.json") and not f.endswith("_semantic_analysis.json")]
    
    if not valid_files:
        return None
        
    # Sort by timestamp in filename (assuming format ..._YYYYMMDD_HHMMSS...)
    # Or just by file modification time as fallback
    # Let's try to extract timestamp
    # Format: {safe_model}_{safe_dataset}_{exp_name}_s{seed}_{timestamp}.json
    # Timestamp is usually the last part before .json
    
    def extract_timestamp(fpath):
        fname = os.path.basename(fpath)
        # Regex to find datetime pattern YYYYMMDD_HHMMSS
        match = re.search(r"(\d{8}_\d{6})", fname)
        if match:
            return match.group(1)
        return ""
    
    # Sort by timestamp string (descending)
    valid_files.sort(key=lambda x: extract_timestamp(x), reverse=True)
    return valid_files[0]

def main():
    parser = argparse.ArgumentParser(description="Collect latest results and append to summary.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="Dataset name")
    parser.add_argument("--output", type=str, default="experiments/results_summary.txt", help="Output summary file")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in experiments/ or root?
    # User said "let's implement a script called 'collect_results.py'".
    # I'll assume it's placed in experiments/ based on context of other scripts.
    # So base_dir is .../experiments
    
    safe_model = args.model.replace('/', '_').replace(' ', '_')
    safe_dataset = args.dataset.replace('/', '_')
    
    # List of experiments to check
    # We can scan directories that have transformation.py
    experiment_names = []
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)):
            if os.path.exists(os.path.join(base_dir, item, "transformation.py")):
                experiment_names.append(item)
    
    experiment_names.sort()
    
    # Prepare Output Lines
    lines = []
    lines.append("")
    lines.append(f"Results for {args.model} on {args.dataset}")
    header = "{:<30} | {:<8} | {:<10} | {:<10} | {}".format("Experiment", "Total", "Accuracy", "Failures", "File")
    lines.append(header)
    lines.append("-" * len(header))
    
    results_found = False
    
    for exp in experiment_names:
        exp_dir = os.path.join(base_dir, exp)
        latest_file = get_latest_result_file(exp_dir, safe_model, safe_dataset)
        
        if latest_file:
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                
                # Check for summary at the end
                if isinstance(data, list) and len(data) > 0 and 'summary' in data[-1]:
                    summary = data[-1]['summary']
                    total = summary.get('total', 0)
                    correct = summary.get('correct', 0)
                    failures = summary.get('failures', 0)
                    acc = summary.get('accuracy', 0.0)
                else:
                    # Manually calculate
                    total = len(data)
                    correct = sum(1 for item in data if item.get('correct') is True)
                    failures = sum(1 for item in data if item.get('extracted') is None or (isinstance(item.get('extracted'), str) and item.get('extracted').startswith("ERROR")))
                    acc = correct / total if total > 0 else 0.0
                
                # Format Percentage
                acc_str = "{:.2%}".format(acc)
                fname = os.path.basename(latest_file)
                
                line = "{:<30} | {:<8} | {:<10} | {:<10} | {}".format(exp, total, acc_str, failures, fname)
                lines.append(line)
                results_found = True
                
            except Exception as e:
                print(f"Error reading {latest_file}: {e}")
        else:
            # Optional: Start print missing experiments?
            # lines.append("{:<30} | {:<8} | {:<10} | {:<10} | {}".format(exp, "-", "-", "-", "No results"))
            pass

    if results_found:
        output_path = args.output
        # If args.output is relative, make it absolute or use as is. 
        # If running from root, experiments/results_summary.txt is fine.
        
        print(f"Appending results to {output_path}...")
        with open(output_path, "a") as f:
            for line in lines:
                f.write(line + "\n")
        
        print("\n".join(lines))
    else:
        print("No results found for any experiment.")

if __name__ == "__main__":
    main()
