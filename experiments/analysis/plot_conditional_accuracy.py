#!/usr/bin/env python3
"""
Plot Conditional Accuracy: Accuracy Given Successful Recovery.
Defined as: (Original Correct / Semantically Recovered) * 100.

Usage:
    python analysis/plot_conditional_accuracy.py --dataset HuggingFaceH4_aime_2024
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
}

TECHNIQUE_ORDER = [
    "baseline",
    "not_not", "opposites", "wrappers",
    "interleaved_context_line", "interleaved_context_word", "interleaved_context_symbol",
    "context_saturation",
    "sentence_reversal", "word_reversal", "split_reversal",
    "rail_fence",
]

MODEL_SHORT_NAMES = {
    "GAIR_LIMO-v2":                                     "LIMO-v2-32B",
    "tiiuae_Falcon-H1R-7B":                             "Falcon-H1R-7B",
    "openai_gpt-oss-120b":                              "GPT-OSS-120B",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-70B":        "DSR1-Llama-70B",
    "Qwen_Qwen3-30B-A3B-Thinking-2507":                 "Qwen3-30B-A3B",
    "gemini-3.1-pro-preview":                           "Gemini 3.1\nPro",
    "claude-opus-4-6":                                  "Claude Opus\n4-6",
}

DATASET_SHORT_NAMES = {
    "HuggingFaceH4_aime_2024":  "AIME 2024",
    "MathArena_aime_2025":      "AIME 2025",
    "MATH_500":                 "MATH 500",
    "MathArena_hmmt_feb_2025":  "HMMT Feb 2025",
}

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#64B5CD", "#CCB974", "#636363"
]

def shorten(name, mapping):
    return mapping.get(name, name)

# ── Data Scanning ───────────────────────────────────────────────────

def scan_conditional_accuracy(experiments_dir, safe_dataset):
    """
    Scan prompt_reconstruction results and calculate Original Correct / Semantic Correct.
    """
    report_base = os.path.join(experiments_dir, "analysis", "prompt_reconstruction", "results")
    data = defaultdict(dict)

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
            for t in TECHNIQUE_ORDER + list(TECHNIQUE_LABELS.keys()):
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
                
                # Accuracy Given Recovery = (Correct & Recovered) / Recovered
                # Since analyze_prompt_recovery.py counts all originally correct samples as recovered,
                # Accuracy Given Recovery = original_correct / semantic_correct
                
                if sem_correct > 0:
                    cond_acc = 100.0 * orig_correct / sem_correct
                else:
                    cond_acc = 0.0

                # Keep latest report per technique/model
                data[matched_technique][model_name] = {
                    'conditional_accuracy': cond_acc,
                    'n_recovered': sem_correct,
                    'n_total': report.get('total_samples', 0)
                }
            except Exception as e:
                print(f"  Warning: could not read {fpath}: {e}")

    # Add baseline (it's 100% since recovered=total and accuracy=accuracy)
    # But wait, baseline accuracy is not in the recovery report. 
    # We should probably fetch it to be thorough, but the user mostly cares about transforms.
    # For now, if someone needs baseline, we'd need to fetch actual accuracy.
    # Actually, for baseline, conditional accuracy is exactly the baseline accuracy.
    
    return data

# ── Plotting ─────────────────────────────────────────────────────────

def plot_conditional_accuracy(dataset_name, technique_data, outdir):
    all_models = set()
    for td in technique_data.values():
        all_models.update(td.keys())
    
    if not all_models:
        print(f"No data for {dataset_name}")
        return

    # Order models by average conditional accuracy
    def _avg_cond_acc(model):
        vals = [technique_data[t][model]['conditional_accuracy'] 
                for t in technique_data if model in technique_data[t]]
        return sum(vals) / len(vals) if vals else 0
    all_models = sorted(all_models, key=_avg_cond_acc, reverse=True)

    ordered_techniques = [t for t in TECHNIQUE_ORDER if t in technique_data]
    for t in sorted(technique_data.keys()):
        if t not in ordered_techniques:
            ordered_techniques.append(t)

    n_techniques = len(ordered_techniques)
    ncols = min(4, n_techniques)
    nrows = (n_techniques + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    if n_techniques == 1: axes = np.array([axes])
    axes = np.atleast_2d(axes)

    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    fig.suptitle(f"Accuracy Given Successful Recovery — {dataset_label}\n(Subset where Semantic Similarity > 90%)", 
                 fontsize=20, fontweight='bold', y=0.98)

    model_colors = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(all_models)}

    for idx, technique in enumerate(ordered_techniques):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        td = technique_data[technique]
        
        subplot_models = [m for m in all_models if m in td]
        if not subplot_models:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            continue

        x = np.arange(len(subplot_models))
        accs = [td[m]['conditional_accuracy'] for m in subplot_models]
        colors = [model_colors[m] for m in subplot_models]
        
        bars = ax.bar(x, accs, 0.65, color=colors, edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                    f"{val:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title(TECHNIQUE_LABELS.get(technique, technique), fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        
        tick_labels = []
        for m in subplot_models:
            short = shorten(m, MODEL_SHORT_NAMES)
            n = td[m]['n_recovered']
            tick_labels.append(f"{short}\n(n={int(n)})")
        
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha='right')
        ax.set_ylabel("Cond. Accuracy (%)", fontsize=11)
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(n_techniques, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"conditional_accuracy_{dataset_name}.pdf")
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

    plot_conditional_accuracy(safe_dataset, data, outdir)

if __name__ == "__main__":
    main()
