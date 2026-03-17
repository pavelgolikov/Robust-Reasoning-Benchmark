#!/usr/bin/env python3
import os
import json
import glob
import time
import argparse
import tiktoken
from tqdm import tqdm
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────

TECHNIQUE_ORDER = [
    "baseline",
    "not_not", "opposites", "wrappers",
    "interleaved_context_line", "interleaved_context_word", "interleaved_context_symbol",
    "context_saturation",
    "sentence_reversal", "word_reversal", "split_reversal",
    "rail_fence",
    "rectangle_perimeter", "snake_vertical", "snake_horizontal",
]

# ── Metrics ──────────────────────────────────────────────────────────

def compute_accuracy(all_results):
    total = 0
    correct = 0
    for r in all_results:
        if r.get("id") is None and "summary" in r:
            continue
        total += 1
        if r.get("correct", False):
            correct += 1
    acc = 100.0 * correct / total if total > 0 else 0
    return correct, total, acc

def compute_avg_length(all_results):
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return 0
    total_tokens = 0
    count = 0
    for r in all_results:
        if r.get("id") is None and "summary" in r:
            continue
        out_text = r.get("output", "")
        if isinstance(out_text, str) and out_text:
            toks = r.get('output_tokens')
            if toks is not None:
                total_tokens += toks
            else:
                try:
                    total_tokens += len(enc.encode(out_text, disallowed_special=()))
                except Exception:
                    pass
            count += 1
    return total_tokens / count if count > 0 else 0

def compute_failure_rate(all_results, total):
    n_failures = 0
    for r in all_results:
        if r.get("id") is None and "summary" in r:
            continue
        if r.get("refusal") is True:
            n_failures += 1
        elif not r.get("correct", False) and r.get("extracted") is None:
            n_failures += 1
    return 100.0 * n_failures / total if total > 0 else 0

# ── Processing ───────────────────────────────────────────────────────

def pick_latest_file(files):
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def scan_and_calculate(experiments_dir, output_base):
    print(f"Scanning results in: {experiments_dir}")
    tasks = []

    # 1. Identify all calculation tasks
    for technique in TECHNIQUE_ORDER:
        technique_dir = os.path.join(experiments_dir, technique, "results")
        if not os.path.isdir(technique_dir):
            continue
            
        for model_name in os.listdir(technique_dir):
            model_dir = os.path.join(technique_dir, model_name)
            if not os.path.isdir(model_dir):
                continue
                
            for dataset_name in os.listdir(model_dir):
                dataset_dir = os.path.join(model_dir, dataset_name)
                if not os.path.isdir(dataset_dir):
                    continue
                
                json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
                # Filter out summaries or other non-result files
                result_files = [f for f in json_files if not (
                    os.path.basename(f).startswith("jobs_") or 
                    os.path.basename(f).startswith("tracking_") or
                    os.path.basename(f).startswith("batch_") or
                    "_summary_" in f or
                    "prompt_recovery" in f
                )]
                
                latest = pick_latest_file(result_files)
                if latest:
                    tasks.append((technique, model_name, dataset_name, latest))

    # 2. Execute and Save
    for technique, model_name, dataset_name, fpath in tqdm(tasks, desc="Calculating lengths"):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "results" in data:
                results = data["results"]
            elif isinstance(data, list):
                results = data
            else:
                continue

            if not results:
                continue

            correct, total, acc = compute_accuracy(results)
            fail_rate = compute_failure_rate(results, total)
            avg_len = compute_avg_length(results)
            
            unique_ids = set(r.get('id') for r in results if r.get('id') is not None)
            n_samples = total / len(unique_ids) if unique_ids else 0

            # Save to summary directory
            summary_folder = os.path.join(output_base, model_name, dataset_name)
            os.makedirs(summary_folder, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            summary_path = os.path.join(summary_folder, f"{technique}_summary_{timestamp}.json")
            
            summary_data = {
                "technique": technique,
                "model": model_name,
                "dataset": dataset_name,
                "accuracy": acc,
                "failure_rate": fail_rate,
                "n_samples": n_samples,
                "length": avg_len,
                "timestamp": timestamp
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Calculate output lengths and accuracy for all results.")
    parser.add_argument("--experiments_dir", type=str, default=None, help="Base experiments directory")
    args = parser.parse_args()

    if args.experiments_dir:
        experiments_dir = args.experiments_dir
    else:
        # Default to parent of current script's directory (assuming script is in analysis/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        experiments_dir = os.path.dirname(script_dir)

    output_base = os.path.join(experiments_dir, "analysis", "output_length", "results")
    
    scan_and_calculate(experiments_dir, output_base)
    print(f"\nProcessing complete. Summaries saved to {output_base}")

if __name__ == "__main__":
    main()
