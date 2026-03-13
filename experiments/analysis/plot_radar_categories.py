#!/usr/bin/env python3
"""
Plot Radar (Spider) Charts by Transformation Category.
Categories:
1. Syntactic Distortions: split_reversal, word_reversal, sentence_reversal
2. Semantic Substitutions: not_not, opposites, wrappers
3. Visual Encoding: rail_fence
4. Contextual Overload: interleaved_context_line, interleaved_context_symbol, interleaved_context_word, context_saturation

Each axis represents the average accuracy in that category (0-100%).
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

from plot_results import scan_results, MODEL_SHORT_NAMES, DATASET_SHORT_NAMES, PALETTE

# ── Configuration ────────────────────────────────────────────────────

CATEGORIES = {
    "Syntactic Distortions": ["split_reversal", "word_reversal", "sentence_reversal"],
    "Semantic Substitutions": ["not_not", "opposites", "wrappers"],
    "Visual Encoding": ["rail_fence"],
    "Contextual Overload": ["interleaved_context_line", "interleaved_context_symbol", "interleaved_context_word", "context_saturation"],
}

CATEGORY_NAMES = list(CATEGORIES.keys())

def plot_radar_charts(dataset_name, data, outdir):
    """
    data[dataset][technique][model] = {accuracy, failure_rate, n_samples, length}
    """
    dataset_data = data.get(dataset_name, {})
    if not dataset_data:
        print(f"No data for dataset {dataset_name}")
        return

    # Aggregate accuracy by category for each model
    model_category_acc = defaultdict(dict)
    all_models = set()
    for tech, models in dataset_data.items():
        all_models.update(models.keys())

    for model in all_models:
        for cat_name, techniques in CATEGORIES.items():
            accs = []
            for t in techniques:
                if t in dataset_data and model in dataset_data[t]:
                    accs.append(dataset_data[t][model]['accuracy'])
            
            if accs:
                model_category_acc[model][cat_name] = sum(accs) / len(accs)
            else:
                model_category_acc[model][cat_name] = 0.0

    # Sort models by average accuracy across transforms (consistent with other plots)
    def _avg_accuracy_global(model):
        accs = [dataset_data[t][model]['accuracy']
                for t in dataset_data if model in dataset_data[t]]
        return sum(accs) / len(accs) if accs else 0
    models_to_plot = sorted(list(all_models), key=_avg_accuracy_global, reverse=True)
    if not models_to_plot:
        return

    num_models = len(models_to_plot)
    ncols = min(4, num_models)
    nrows = (num_models + ncols - 1) // ncols

    fig = plt.figure(figsize=(5.5 * ncols, 6 * nrows))
    dataset_label = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
    fig.suptitle(f"Model Performance by Transformation Category — {dataset_label}", 
                 fontsize=22, fontweight='bold', y=0.98)

    angles = np.linspace(0, 2 * np.pi, len(CATEGORY_NAMES), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    for i, model in enumerate(models_to_plot):
        ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
        
        values = [model_category_acc[model][cat] for cat in CATEGORY_NAMES]
        values += values[:1]  # close the loop
        
        color = PALETTE[i % len(PALETTE)]
        
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.25)
        
        # Set category labels
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Fix axis to 0-100
        ax.set_ylim(0, 100)
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=8)
        
        plt.xticks(angles[:-1], CATEGORY_NAMES, color='black', size=9, fontweight='bold')
        
        # Title for subplot
        model_label = MODEL_SHORT_NAMES.get(model, model).replace('\n', ' ')
        ax.set_title(model_label, size=14, fontweight='bold', pad=20)
        
        # Add a light grid
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"radar_categories_{dataset_name}.pdf")
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
    data = scan_results(experiments_dir, aggregate=False)
    
    safe_dataset = args.dataset.replace('/', '_')
    plot_radar_charts(safe_dataset, data, outdir)

if __name__ == "__main__":
    main()
