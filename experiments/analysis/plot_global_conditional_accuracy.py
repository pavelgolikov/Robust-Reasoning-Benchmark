#!/usr/bin/env python3
"""
Plot Global Conditional Accuracy: Baseline vs. Micro-Averaged Conditional Accuracy.
Formula: Sum(solved in all traps) / Sum(recovered in all traps) * 100.
Excludes control transforms (baseline, context_saturation).
"""

import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add current directory to path to import helpers
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from plot_conditional_accuracy import scan_conditional_accuracy
from plot_results import scan_results, MODEL_SHORT_NAMES, DATASET_SHORT_NAMES, PALETTE

def plot_global_conditional_accuracy(dataset_name, cond_data, experiments_dir, outdir):
    """
    cond_data[model][technique] = {conditional_accuracy, n_recovered, n_total, n_solved}
    """
    # 1. Fetch baseline accuracy from main results
    print(f"Fetching baseline accuracy for {dataset_name}...")
    accuracy_data_all = scan_results(experiments_dir, aggregate=False)
    accuracy_data = accuracy_data_all.get(dataset_name, {})
    
    # 2. Calculate Micro-Averaged Global Conditional Accuracy
    # Exclude control transforms
    exclude_techs = ["baseline", "context_saturation"]
    
    plot_data = [] # List of (model, baseline_acc, global_cond_acc)
    
    for model_name, techs in cond_data.items():
        sum_solved = 0
        sum_recovered = 0
        
        for t, metrics in techs.items():
            if t in exclude_techs:
                continue
            
            sum_solved += metrics.get('n_solved', 0)
            sum_recovered += metrics.get('n_recovered', 0)
            
        if sum_recovered > 0:
            global_cond_acc = 100.0 * sum_solved / sum_recovered
        else:
            global_cond_acc = 0.0
            
        # Get baseline accuracy
        baseline_acc = 0.0
        if "baseline" in accuracy_data and model_name in accuracy_data["baseline"]:
            baseline_acc = accuracy_data["baseline"][model_name]['accuracy']
            
        plot_data.append({
            'model': model_name,
            'baseline': baseline_acc,
            'global_cond': global_cond_acc
        })
        
    if not plot_data:
        print("No plot data collected.")
        return

    # Sort models by global conditional accuracy (highest first)
    plot_data = sorted(plot_data, key=lambda x: x['global_cond'], reverse=True)
    
    # 3. Plotting
    models = [d['model'] for d in plot_data]
    short_models = [MODEL_SHORT_NAMES.get(m, m).replace('\n', ' ') for m in models]
    baseline_accs = [d['baseline'] for d in plot_data]
    global_cond_accs = [d['global_cond'] for d in plot_data]
    
    x = np.arange(len(models))
    width = 0.35  # width of the bars
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Solid blue for baseline
    rects1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline Accuracy', 
                    color='#4C72B0', edgecolor='white', linewidth=0.5)
    
    # Solid orange for global conditional (micro-averaged)
    rects2 = ax.bar(x + width/2, global_cond_accs, width, label='Global Conditional Accuracy', 
                    color='#DD8452', edgecolor='white', linewidth=0.5)
    
    # Add labels, title and custom x-axis tick labels, etc.
    dataset_label = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
    ax.set_title(f"Global Reasoning Stability — {dataset_label}\nMicro-Averaged Accuracy Given Successful Recovery (Excl. Controls)", 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_models, rotation=25, ha='right', fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f9f9f9')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"global_conditional_accuracy_{dataset_name}.pdf")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4_aime_2024")
    args = parser.parse_args()
    
    experiments_dir = os.path.dirname(script_dir)
    outdir = os.path.join(experiments_dir, "analysis", "plots")
    
    safe_dataset = args.dataset.replace('/', '_')
    cond_data = scan_conditional_accuracy(experiments_dir, safe_dataset)
    
    if not cond_data:
        print("No conditional data found.")
        return
        
    plot_global_conditional_accuracy(safe_dataset, cond_data, experiments_dir, outdir)

if __name__ == "__main__":
    main()
