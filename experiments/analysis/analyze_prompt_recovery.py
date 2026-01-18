
import json
import argparse
import os
import re
import glob
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

def normalize_text(text: str) -> str:
    """Basic normalization: remove latex, extra whitespace."""
    if not text:
        return ""
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
    text = text.replace('$', '').replace('\\', '')
    return " ".join(text.split())

def make_windows(tokens: List[str], window_size: int, step_size: int = 10) -> List[str]:
    """Create sliding windows of text from tokens."""
    windows = []
    if not tokens:
        return []
    if len(tokens) <= window_size:
        return [" ".join(tokens)]
    
    for i in range(0, len(tokens) - window_size + 1, step_size):
        window = tokens[i : i + window_size]
        windows.append(" ".join(window))
    
    if len(tokens) > window_size:
        last_window = tokens[-window_size:]
        windows.append(" ".join(last_window))
        
    return list(set(windows))

def find_latest_result(experiment_name: str, base_dir: str = "experiments") -> str:
    """Finds the latest JSON result file for a given experiment."""
    # Handle baseline case separately if needed, or assume standard structure
    # Standard structure: experiments/{name}/results/GAIR_LIMO-v2_{name}_s42_{timestamp}.json
    
    # Try standard pattern
    results_dir = os.path.join(base_dir, experiment_name, "results")
    if not os.path.exists(results_dir):
        # Fallback for baseline if it's just in experiments/baseline/results/ without complicated name match?
        # Or maybe the user didn't create the folder yet.
        return None
        
    # Pattern matching
    # We want files that look like results JSONs.
    files = glob.glob(os.path.join(results_dir, "*.json"))
    # Filter out *semantic_analysis.json or _raw.json if any exist
    files = [f for f in files if not f.endswith("_semantic_analysis.json") and "semantic" not in f and not f.endswith("_raw.json")]
    
    if not files:
        return None
        
    # Sort by modification time (or filename timestamp if robust, but mtime is easier)
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def analyze_single_file(result_file: str, model: SentenceTransformer, args) -> Dict[str, Any]:
    if args.dry:
        return {
            "source_file": result_file,
            "total_samples": 100,
            "original_correct": 10,
            "semantic_correct": 20,
            "recovered_cases": [{"id": 0, "score": 0.99, "target": "DRY RUN", "best_window": "DRY RUN"}],
            "original_accuracy": 0.1,
            "semantic_accuracy": 0.2,
            "note": "DRY RUN - MOCK DATA"
        }

    with open(result_file, 'r') as f:
        data = json.load(f)

    total = len(data)
    summary = {
        "source_file": result_file,
        "total_samples": total,
        "original_correct": 0,
        "semantic_correct": 0,
        "recovered_cases": []
    }

    # Iterate with progress bar
    for entry in tqdm(data, desc=f"Analyzing {os.path.basename(result_file)}", leave=False):
        is_orig_correct = entry.get('correct', False)
        if is_orig_correct:
            summary["original_correct"] += 1
            summary["semantic_correct"] += 1
            continue 
            
        target_text = entry.get('unmodified_original', '')
        model_output = entry.get('output', '')
        
        norm_target = normalize_text(target_text)
        target_tokens = norm_target.split()
        target_len = len(target_tokens)
        
        norm_output = normalize_text(model_output)
        output_tokens = norm_output.split()
        
        if not norm_target or not norm_output:
            continue

        window_sizes = [int(target_len * 0.8), target_len, int(target_len * 1.2)]
        
        all_windows = []
        for w_size in window_sizes:
            if w_size < 1: w_size = 1
            all_windows.extend(make_windows(output_tokens, window_size=w_size, step_size=args.step_size))
        
        if not all_windows:
            continue
            
        target_embedding = model.encode(norm_target, convert_to_tensor=True, show_progress_bar=False)
        window_embeddings = model.encode(all_windows, convert_to_tensor=True, show_progress_bar=False)
        
        cosine_scores = util.cos_sim(target_embedding, window_embeddings)[0]
        
        best_idx = int(cosine_scores.argmax())
        max_score = float(cosine_scores[best_idx])
        
        if max_score >= args.threshold:
            summary["semantic_correct"] += 1
            summary["recovered_cases"].append({
                "id": entry.get('id'),
                "score": max_score,
                "target": norm_target,
                "best_window": all_windows[best_idx]
            })

    summary["original_accuracy"] = summary["original_correct"] / total if total > 0 else 0
    summary["semantic_accuracy"] = summary["semantic_correct"] / total if total > 0 else 0
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs='+', required=True, help="List of experiment names (e.g. 'word_reversal' 'baseline') or 'all'")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--dry", action="store_true", help="Dry run: print discovered files without processing")
    args = parser.parse_args()

    # Determine techniques to process
    # Correctly locate 'experiments' dir relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__)) # experiments/analysis
    experiments_dir = os.path.dirname(script_dir) # experiments
    
    # Robust way: Use the script location as anchor
    base_dir = experiments_dir
    
    output_dir = os.path.join(base_dir, "analysis", "results")
    summary_file = os.path.join(base_dir, "analysis", "prompt_recovery_analysis.txt")
    
    # Always create output directory
    os.makedirs(output_dir, exist_ok=True)

    techniques = args.names
    if "all" in techniques:
        # Auto-discover directories in experiments that have a 'results' subdir
        techniques = []
        excluded_dirs = {"analysis", "variables", "__pycache__", "baseline"}  # Exclude non-experiment dirs
        for d in os.listdir(base_dir):
            if d in excluded_dirs:
                continue
            dir_path = os.path.join(base_dir, d)
            results_path = os.path.join(dir_path, "results")
            if os.path.isdir(dir_path) and os.path.exists(results_path):
                # Double check that there is actually a result file inside
                if find_latest_result(d, base_dir):
                    techniques.append(d)
        techniques.sort()

    print(f"Techniques to analyze: {techniques}")
    print(f"Base Directory: {base_dir}")
    print(f"Output Directory: {output_dir}")
    
    if args.dry:
        print("\n--- DRY RUN: MOCKING ANALYSIS ---")
        model = None
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer model on {device}...")
        model = SentenceTransformer(args.model_name, device=device)

    # Store results for summary table
    table_rows = []

    for name in techniques:
        print(f"\nProcessing: {name}")
        latest_file = find_latest_result(name, base_dir)
        
        if not latest_file:
            print(f"  No result file found for {name}. Skipping.")
            continue
            
        print(f"  Latest file: {os.path.basename(latest_file)}")
        
        summary = analyze_single_file(latest_file, model, args)
        
        # Save detailed report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{name}_prompt_recovery_{timestamp}.json"
        
        if args.dry:
             output_filename = f"{name}_prompt_recovery_DRYRUN.json"
             
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved report to: {output_path}")
        
        # Collect stats
        row = {
            "name": name,
            "total": summary["total_samples"],
            "orig_acc": summary["original_accuracy"],
            "sem_acc": summary["semantic_accuracy"],
            "recovered": len(summary["recovered_cases"]),
            "file": os.path.basename(latest_file)
        }
        table_rows.append(row)

    # Generate Summary Table
    header = f"{'Experiment':<30} | {'Total':<8} | {'Orig Acc':<10} | {'Sem Acc':<10} | {'Recovered':<10} | {'File'}"
    divider = "-" * len(header)
    
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    summary_text = f"\n\nAnalysis Run: {timestamp_str} (DRY RUN)\n" if args.dry else f"\n\nAnalysis Run: {timestamp_str}\n"
    summary_text += header + "\n" + divider + "\n"
    
    print("\n" + header)
    print(divider)
    
    for row in table_rows:
        line = f"{row['name']:<30} | {row['total']:<8} | {row['orig_acc']:<10.2%} | {row['sem_acc']:<10.2%} | {row['recovered']:<10} | {row['file']}"
        summary_text += line + "\n"
        print(line)

    # Append to summary file
    with open(summary_file, "a") as f:
        f.write(summary_text)
    print(f"\nSummary appended to: {summary_file}")

if __name__ == "__main__":
    main()

