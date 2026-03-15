#!/usr/bin/env python3
import argparse
import os
import sys
import json
from collections import defaultdict

# Add current directory to path to import helpers
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from plot_results import scan_results, TECHNIQUE_ORDER, TECHNIQUE_LABELS, scan_recovery_results
from plot_conditional_accuracy import scan_conditional_accuracy

def aggregate_data(dataset_name, experiments_dir):
    print(f"Aggregating data for {dataset_name}...")
    
    # 1. Scan main results (Accuracy, length, etc.)
    all_data = scan_results(experiments_dir, aggregate=True)
    dataset_results = all_data.get(dataset_name, {})
    
    # 2. Scan recovery results
    recovery_data = scan_recovery_results(experiments_dir, TECHNIQUE_ORDER, dataset_name)
    
    # 3. Scan conditional accuracy
    cond_data = scan_conditional_accuracy(experiments_dir, dataset_name)
    
    return dataset_results, recovery_data, cond_data

def format_summary(dataset_name, dataset_results, recovery_data, cond_data):
    lines = []
    lines.append("=" * 80)
    lines.append(f"RESULTS SUMMARY: {dataset_name}")
    lines.append("=" * 80)
    lines.append("")
    
    # --- Part 1: Main Results (Accuracy & Length) ---
    lines.append("--- PART 1: ACCURACY & OUTPUT LENGTH ---")
    
    # Get all models
    all_models = set()
    for tech in dataset_results:
        all_models.update(dataset_results[tech].keys())
    all_models = sorted(list(all_models))
    
    for tech in TECHNIQUE_ORDER:
        if tech not in dataset_results:
            continue
        
        tech_label = TECHNIQUE_LABELS.get(tech, tech)
        lines.append(f"\n[Technique: {tech_label}] ({tech})")
        lines.append(f"{'Model':<40} | {'Acc (%)':<10} | {'Fail (%)':<10} | {'Length':<10} | {'Samples'}")
        lines.append("-" * 85)
        
        for model in all_models:
            if model in dataset_results[tech]:
                m_data = dataset_results[tech][model]
                acc = m_data.get('accuracy', 0)
                fail = m_data.get('failure_rate', 0)
                length = m_data.get('length', 0)
                samples = m_data.get('n_samples', 0)
                lines.append(f"{model:<40} | {acc:<10.2f} | {fail:<10.2f} | {length:<10.1f} | {samples}")
    
    lines.append("\n" + "=" * 80)
    
    # --- Part 2: Recovery Rates ---
    lines.append("\n--- PART 2: PROMPT RECOVERY RATES ---")
    lines.append(f"{'Model':<40} | {'Technique':<30} | {'Recovery Rate (%)'}")
    lines.append("-" * 85)
    
    for model in all_models:
        for tech in TECHNIQUE_ORDER:
            if tech in recovery_data and model in recovery_data[tech]:
                rate = recovery_data[tech][model].get('recovery_rate', 0)
                lines.append(f"{model:<40} | {tech:<30} | {rate:<10.2f}")
    
    lines.append("\n" + "=" * 80)
    
    # --- Part 3: Conditional Accuracy ---
    lines.append("\n--- PART 3: CONDITIONAL ACCURACY (ACCURACY GIVEN RECOVERY) ---")
    lines.append(f"{'Model':<40} | {'Technique':<30} | {'Cond. Acc (%)'} | {'Recovered'}")
    lines.append("-" * 85)
    
    for model in all_models:
        if model in cond_data:
            for tech in TECHNIQUE_ORDER:
                if tech in cond_data[model]:
                    c_data = cond_data[model][tech]
                    acc = c_data.get('conditional_accuracy', 0)
                    recovered = c_data.get('n_recovered', 0)
                    lines.append(f"{model:<40} | {tech:<30} | {acc:<10.2f} | {recovered}")
                    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4_aime_2024")
    parser.add_argument("--output", type=str, default="results_summary.txt")
    args = parser.parse_args()

    experiments_dir = os.path.dirname(script_dir)
    safe_dataset = args.dataset.replace('/', '_')
    
    dataset_results, recovery_data, cond_data = aggregate_data(safe_dataset, experiments_dir)
    
    if not dataset_results:
        print(f"No results found for {safe_dataset}")
        return

    summary_text = format_summary(safe_dataset, dataset_results, recovery_data, cond_data)
    
    output_path = os.path.join(script_dir, args.output)
    with open(output_path, 'w') as f:
        f.write(summary_text)
    
    print(f"\nAggregation complete. Summary saved to: {output_path}")

if __name__ == "__main__":
    main()
