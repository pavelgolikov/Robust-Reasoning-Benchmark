#!/usr/bin/env python3
"""
Plot "Over/Under Thinking": Delta Output Length vs. Delta Accuracy Scatter Plot.
X-axis: Transform Length - Baseline Length
Y-axis: Transform Accuracy - Baseline Accuracy

Specifically for open-source reasoning models.
"""

import argparse
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add current directory to path to import scan_results
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from plot_results import scan_results, TECHNIQUE_LABELS, MODEL_SHORT_NAMES, DATASET_SHORT_NAMES, PALETTE

# ── Configuration ────────────────────────────────────────────────────

OPEN_SOURCE_MODELS = [
    "GAIR_LIMO-v2",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-70B",
    "Qwen_Qwen3-30B-A3B-Thinking-2507",
    "tiiuae_Falcon-H1R-7B",
    "openai_gpt-oss-120b",
]

# Markers for different transformations to distinguish them in the scatter plot
TECHNIQUE_MARKERS = {
    "opposites": "o",
    "not_not": "s",
    "wrappers": "D",
    "split_reversal": "^",
    "word_reversal": "v",
    "sentence_reversal": "<",
    "rail_fence": ">",
    "interleaved_context_line": "p",
    "interleaved_context_word": "*",
    "interleaved_context_symbol": "H",
    "context_saturation": "X",
}

def plot_over_under_thinking(dataset_name, data, outdir):
    """
    data[dataset][technique][model] = {accuracy, failure_rate, n_samples, length}
    """
    dataset_data = data.get(dataset_name, {})
    if not dataset_data:
        print(f"No data for dataset {dataset_name}")
        return

    # 1. Identify baseline for each model
    baselines = dataset_data.get("baseline", {})
    
    # 2. Collect points (delta_length, delta_accuracy)
    # Collect as: model -> list of (dx, dy, technique)
    model_points = defaultdict(list)
    
    for technique, models in dataset_data.items():
        if technique == "baseline":
            continue
        
        for model, results in models.items():
            if model not in OPEN_SOURCE_MODELS:
                continue
            
            if model not in baselines:
                continue
            
            b_acc = baselines[model]['accuracy']
            b_len = baselines[model]['length']
            
            dx = results['length'] - b_len
            dy = results['accuracy'] - b_acc
            
            model_points[model].append((dx, dy, technique))

    if not model_points:
        print(f"No points collected for {dataset_name} (check model names and baseline availability)")
        return

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw quadrants
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)

    # Colors for models
    model_colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(OPEN_SOURCE_MODELS)}

    # Plot points
    legend_handles = []
    
    for model in OPEN_SOURCE_MODELS:
        if model not in model_points:
            continue
            
        points = model_points[model]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        c = model_colors[model]
        label = MODEL_SHORT_NAMES.get(model, model).replace('\n', ' ')
        
        # Plot each point with technique-specific marker
        for dx, dy, tech in points:
            marker = TECHNIQUE_MARKERS.get(tech, 'x')
            ax.scatter(dx, dy, s=150, color=c, marker=marker, alpha=0.7, edgecolors='black', linewidth=0.5)
            
        # Dummy plot for model legend
        h = ax.scatter([], [], s=100, color=c, marker='o', label=label)
        legend_handles.append(h)

    # Create technique legend
    tech_legend_handles = []
    unique_techs = sorted(list(set(p[2] for m in model_points for p in model_points[m])))
    for tech in unique_techs:
        marker = TECHNIQUE_MARKERS.get(tech, 'x')
        label = TECHNIQUE_LABELS.get(tech, tech)
        h = ax.scatter([], [], s=100, color='grey', marker=marker, label=label)
        tech_legend_handles.append(h)

    # Labels and Titles
    dataset_label = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
    ax.set_title(f"Over/Under Thinking: Reasoning Effort vs. Accuracy — {dataset_label}\nOpen-Source Reasoning Models", 
                 fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel("Change in Reasoning Length (Tokens vs. Baseline)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Change in Accuracy (% vs. Baseline)", fontsize=14, fontweight='bold')

    # Formatting
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_facecolor('#f9f9f9')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legends
    model_legend = ax.legend(handles=legend_handles, title="Models", loc='upper left', fontsize=10, frameon=True)
    ax.add_artist(model_legend)
    ax.legend(handles=tech_legend_handles, title="Transformations", loc='lower right', fontsize=9, ncol=2, frameon=True)

    # # Add annotations for the "Over Thinking" zone
    # ax.annotate("OVER-THINKING\n(High Effort, Low Accuracy)", 
    #             xy=(15000, -20), xytext=(12000, -40),
    #             arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.5),
    #             fontsize=16, fontweight='bold', color='darkred', ha='center',
    #             bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.1, ec="red"))

    # # Add annotations for the "Under Thinking" zone
    # ax.annotate("UNDER-THINKING\n(Low Effort, Low Accuracy)", 
    #             xy=(-100, -20), xytext=(-2000, -40),
    #             arrowprops=dict(facecolor='orange', shrink=0.05, alpha=0.5),
    #             fontsize=16, fontweight='bold', color='darkorange', ha='center',
    #             bbox=dict(boxstyle="round,pad=0.3", fc="orange", alpha=0.1, ec="orange"))

    plt.tight_layout()
    
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"over_under_thinking_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4_aime_2024")
    args = parser.parse_args()

    experiments_dir = os.path.dirname(script_dir)
    outdir = os.path.join(experiments_dir, "analysis", "plots")

    print(f"Scanning results in {experiments_dir}...")
    data = scan_results(experiments_dir, aggregate=False, calc_length=True, force_scan=False)
    
    safe_dataset = args.dataset.replace('/', '_')
    plot_over_under_thinking(safe_dataset, data, outdir)

if __name__ == "__main__":
    main()
