import os
import json
import argparse
from collections import defaultdict

def normalize(text):
    if not text:
        return ""
    # Remove all whitespace
    return "".join(text.split())

def analyze_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 0
    
    total = 0
    correct = 0
    
    for entry in data:
        total += 1
        output = entry.get('output', '')
        # unmodified_original corresponds to the ground truth problem statement
        original = entry.get('unmodified_original', '')
        
        norm_output = normalize(output)
        norm_original = normalize(original)
        
        # Check if the normalized original is contained in the normalized output
        if norm_original and norm_original in norm_output:
            correct += 1
            
    return correct, total

def main():
    parser = argparse.ArgumentParser(description="Analyze decoding accuracy from raw results.")
    parser.add_argument("--dir", type=str, default=".", help="Base directory to search for results")
    args = parser.parse_args()
    
    # Walk through directory to find files
    results_files = []
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith("_raw.json"):
                 results_files.append(os.path.join(root, file))
    
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    print(f"Found {len(results_files)} raw result files.")
    
    for filepath in results_files:
        # Determine experiment name from directory structure
        # Assumes structure: .../experiment_name/results/filename.json
        path_parts = filepath.split(os.sep)
        if 'results' in path_parts:
            exp_index = path_parts.index('results') - 1
            if exp_index >= 0:
                exp_name = path_parts[exp_index]
            else:
                exp_name = "unknown"
        else:
            exp_name = os.path.basename(filepath).split('_raw.json')[0]

        c, t = analyze_file(filepath)
        stats[exp_name]["correct"] += c
        stats[exp_name]["total"] += t
        # print(f"Processed {filepath}: {c}/{t}")

    print("\n" + "="*60)
    print(f"{'Experiment':<30} | {'Accuracy':<10} | {'Count':<10}")
    print("-" * 60)
    
    grand_correct = 0
    grand_total = 0
    
    for exp_name, data in sorted(stats.items()):
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        print(f"{exp_name:<30} | {acc:.2%}    | {data['correct']}/{data['total']}")
        grand_correct += data["correct"]
        grand_total += data["total"]
        
    print("-" * 60)
    total_acc = grand_correct / grand_total if grand_total > 0 else 0
    print(f"{'TOTAL':<30} | {total_acc:.2%}    | {grand_correct}/{grand_total}")
    print("="*60)

if __name__ == "__main__":
    main()
