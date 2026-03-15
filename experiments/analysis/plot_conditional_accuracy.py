#!/usr/bin/env python3
"""
Plot Conditional Accuracy: Accuracy Given Successful Recovery.
Refactored: One subplot per model, transformations on the X-axis.
"""

import argparse
import json
import os
import glob
import sys
import time
from collections import defaultdict
import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Configuration (mirrored from plot_results.py) ────────────────────

TECHNIQUE_LABELS = {
    "baseline":                     "Baseline",
    "opposites":                    "Opposites",
    "not_not":                      "Not-Not",
    "wrappers":                     "Wrappers",
    "split_reversal":               "Split Reversal",
    "word_reversal":                "Word Reversal",
    "sentence_reversal":            "Sentence Reversal",
    "rail_fence":                   "Rail Fence",
    "interleaved_context_line":     "Interleave (Line)",
    "interleaved_context_word":     "Interleave (Word)",
    "interleaved_context_symbol":   "Interleave (Symbol)",
    "context_saturation":           "Context Saturation",
    "rectangle_perimeter":          "Rectangle Perimeter",
    "snake_vertical":               "Snake (Vertical)",
    "snake_horizontal":             "Snake (Horizontal)",
}

TECHNIQUE_ORDER = [
    "baseline",
    "not_not", "opposites", "wrappers",
    "interleaved_context_line", "interleaved_context_word", "interleaved_context_symbol",
    "context_saturation",
    "sentence_reversal", "word_reversal", "split_reversal",
    "rail_fence",
    "rectangle_perimeter", "snake_vertical", "snake_horizontal",
]

MODEL_SHORT_NAMES = {
    "GAIR_LIMO-v2":                                     "LIMO-v2-32B",
    "tiiuae_Falcon-H1R-7B":                             "Falcon-H1R-7B",
    "openai_gpt-oss-120b":                              "GPT-OSS-120B",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-70B":        "DSR1-Llama-70B",
    "Qwen_Qwen3-30B-A3B-Thinking-2507":                 "Qwen3-30B-A3B",
    "gemini-3.1-pro-preview":                           "Gemini 3.1 Pro",
    "claude-opus-4-6":                                  "Claude Opus 4-6",
}

DATASET_SHORT_NAMES = {
    "HuggingFaceH4_aime_2024":  "AIME 2024",
    "MathArena_aime_2025":      "AIME 2025",
    "MATH_500":                 "MATH 500",
    "MathArena_hmmt_feb_2025":  "HMMT Feb 2025",
}

PALETTE = [
    "#4C72B0",  # Steel blue
    "#DD8452",  # Sandy brown
    "#55A868",  # Muted green
    "#C44E52",  # Brick red
    "#8172B3",  # Soft purple
    "#937860",  # Dusty brown
    "#DA8BC3",  # Soft pink
    "#64B5CD",  # Teal
    "#CCB974",  # Gold
    "#636363",  # Grey
    "#764978",  # Deep violet
    "#006400",  # Dark green
    "#8B0000",  # Dark red
]

def shorten(name, mapping):
    return mapping.get(name, name)

# ── Data Scanning ───────────────────────────────────────────────────

def scan_conditional_accuracy(experiments_dir, safe_dataset):
    report_base = os.path.join(experiments_dir, "analysis", "prompt_reconstruction", "results")
    data = defaultdict(lambda: defaultdict(dict))

    if not os.path.isdir(report_base):
        print(f"Error: Prompt reconstruction results not found at {report_base}")
        return data

    model_dirs = sorted(os.listdir(report_base))
    for model_name in tqdm.tqdm(model_dirs, desc="Scanning recovery data"):
        model_dataset_dir = os.path.join(report_base, model_name, safe_dataset)
        if not os.path.isdir(model_dataset_dir):
            continue

        for fname in sorted(os.listdir(model_dataset_dir)):
            if not fname.endswith('.json') or 'prompt_recovery' not in fname:
                continue

            matched_technique = None
            for t in TECHNIQUE_ORDER:
                if fname.startswith(t + "_prompt_recovery"):
                    matched_technique = t
                    break

            if not matched_technique:
                continue

            fpath = os.path.join(model_dataset_dir, fname)
            try:
                with open(fpath) as f:
                    report = json.load(f)

                orig_correct = report.get('original_correct', 0)
                sem_correct = report.get('semantic_correct', 0)
                
                if sem_correct > 0:
                    cond_acc = 100.0 * orig_correct / sem_correct
                else:
                    cond_acc = 0.0

                # model -> technique -> data
                data[model_name][matched_technique] = {
                    'conditional_accuracy': cond_acc,
                    'n_recovered': sem_correct,
                    'n_total': report.get('total_samples', 0),
                    'n_solved': orig_correct
                }
            except Exception as e:
                print(f"  Warning: could not read {fpath}: {e}")

    return data

# ── Plotting ─────────────────────────────────────────────────────────

def plot_conditional_accuracy_per_model(dataset_name, model_data, outdir, experiments_dir):
    all_models = list(model_data.keys())
    if not all_models:
        print(f"No data for {dataset_name}")
        return

    # Sort models by global average accuracy to match other plots
    try:
        sys.path.append(os.path.join(experiments_dir, "analysis"))
        from plot_results import scan_results
        accuracy_data_all = scan_results(experiments_dir, aggregate=False)
        accuracy_data = accuracy_data_all.get(dataset_name, {})

        def _avg_accuracy(model):
            accs = [accuracy_data[t][model]['accuracy']
                    for t in accuracy_data if model in accuracy_data[t]]
            return sum(accs) / len(accs) if accs else 0

        all_models = sorted(all_models, key=_avg_accuracy, reverse=True)
    except Exception as e:
        print(f"Warning: sorting by accuracy failed: {e}. Falling back to alphabetical.")
        all_models = sorted(all_models)

    n_models = len(all_models)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    if n_models == 1: axes = np.array([axes])
    axes = np.atleast_2d(axes)

    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    fig.suptitle(f"Accuracy Given Successful Recovery (per Model) — {dataset_label}\n(Subset where Semantic Similarity > 90%)", 
                 fontsize=22, fontweight='bold', y=0.98)

    # Consistent colors per technique (anchored to TECHNIQUE_ORDER)
    tech_colors = {}
    for i, t in enumerate(TECHNIQUE_ORDER):
        tech_colors[t] = PALETTE[i % len(PALETTE)]

    for idx, model_name in enumerate(all_models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        
        m_techs = model_data[model_name]

        # Inject baseline and context_saturation from main results if missing
        # Their conditional accuracy = actual accuracy since they see decoded prompt
        for t in ["baseline", "context_saturation"]:
            if t in accuracy_data and model_name in accuracy_data[t]:
                acc = accuracy_data[t][model_name]['accuracy']
                n_samples = accuracy_data[t][model_name]['n_samples']
                # Always overwrite or fill for these two
                m_techs[t] = {
                    'conditional_accuracy': acc,
                    'n_recovered': n_samples,
                    'n_total': n_samples,
                    'n_solved_pct': acc  # Store the exact percentage to avoid rounding drift
                }
        
        # Keep consistent technique order
        plot_techs = [t for t in TECHNIQUE_ORDER if t in m_techs]
        
        x = np.arange(len(plot_techs))
        bar_width = 0.65
        accs = [m_techs[t]['conditional_accuracy'] for t in plot_techs]
        colors = [tech_colors[t] for t in plot_techs]
        
        bars = ax.bar(x, accs, bar_width, color=colors, edgecolor='black', linewidth=0.5)

        for bar, val, tech in zip(bars, accs, plot_techs):
            n = m_techs[tech]['n_recovered']
            
            if 'n_solved_pct' in m_techs[tech]:
                solve_pct = m_techs[tech]['n_solved_pct']
            else:
                n_solved = m_techs[tech].get('n_solved', 0)
                n_total = m_techs[tech].get('n_total', 0)
                solve_pct = 100.0 * n_solved / n_total if n_total > 0 else 0.0
            
            # Label: Cond% \n (Solve%)
            # Using .1f for both ensures they look identical for baseline/saturation
            label_text = f"{val:.1f}%\n({solve_pct:.1f}%)"
            
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                    label_text, ha='center', va='bottom', fontsize=7, fontweight='bold')

        title = shorten(model_name, MODEL_SHORT_NAMES).replace('\n', ' ')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(x)
        
        # Simple x-tick labels: Just the technique name
        ax.set_xticklabels([shorten(t, TECHNIQUE_LABELS) for t in plot_techs], 
                           fontsize=9, rotation=45, ha='right')
        ax.set_ylabel("Cond. Accuracy (%)", fontsize=11)
        ax.set_ylim(0, 125)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"conditional_accuracy_by_model_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4_aime_2024")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.dirname(script_dir)
    outdir = os.path.join(experiments_dir, "analysis", "plots")

    safe_dataset = args.dataset.replace('/', '_')
    data = scan_conditional_accuracy(experiments_dir, safe_dataset)
    
    if not data:
        print("No data found.")
        return

    plot_conditional_accuracy_per_model(safe_dataset, data, outdir, experiments_dir)

if __name__ == "__main__":
    main()
