#!/usr/bin/env python3
"""
Plot evaluation results as bar-chart grids.

For each dataset found in the results, produces one large figure with one
subplot per transformation.  X-axis = models, Y-axis = accuracy (%).
Bars are color-coded per model, with the accuracy value annotated on top.

Usage (from experiments/ directory):
    python analysis/plot_results.py                       # auto-discover everything
    python analysis/plot_results.py --dataset HuggingFaceH4_aime_2024
    python analysis/plot_results.py --outdir /tmp/plots
"""

import argparse
import json
import os
import glob
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Configuration ────────────────────────────────────────────────────

# Techniques whose `results/` directories to scan
TECHNIQUES = [
    "baseline",
    "opposites",
    "not_not",
    "wrappers",
    "split_reversal",
    "word_reversal",
    "sentence_reversal",
    "rail_fence",
    "interleaved_context_line",
    "interleaved_context_word",
    "interleaved_context_symbol",
]

# Pretty labels for subplot titles
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
}

# Shorten model names for x-axis
MODEL_SHORT_NAMES = {
    "GAIR_LIMO-v2":                                     "LIMO-v2\n(32B)",
    "HAIR_LIMO-v2":                                     "LIMO-v2\n(32B)",   # typo alias
    "tiiuae_Falcon-H1R-7B":                             "Falcon-H1R\n(7B)",
    "openai_gpt-oss-120b":                              "GPT-OSS\n(120B)",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-70B":        "DSR1-Llama\n(70B)",
    "Qwen_Qwen3-235B-A22B-Thinking-2507":               "Qwen3-235B",
    "Qwen_Qwen3-30B-A3B-Thinking-2507":                 "Qwen3-30B",
}

DATASET_SHORT_NAMES = {
    "HuggingFaceH4_aime_2024":  "AIME 2024",
    "MathArena_aime_2025":      "AIME 2025",
    "MATH_500":                 "MATH 500",
    "MathArena_hmmt_feb_2025":  "HMMT Feb 2025",
}

# ── Helpers ──────────────────────────────────────────────────────────

def compute_accuracy(results_list):
    """Given a list of result dicts, return (correct, total, accuracy%)."""
    total = len(results_list)
    if total == 0:
        return 0, 0, 0.0
    correct = sum(1 for r in results_list if r.get("correct", False))
    return correct, total, 100.0 * correct / total


def pick_latest_file(json_files):
    """From a list of json file paths, pick the one with the latest timestamp in name."""
    if not json_files:
        return None
    # Files have timestamps like _20260303_223355.json — sort lexicographically (works for YYYYMMDD_HHMMSS)
    return sorted(json_files)[-1]


def scan_results(experiments_dir):
    """
    Scan all technique directories under experiments_dir.
    Returns:
        data[dataset][technique][model] = accuracy_pct
    """
    data = defaultdict(lambda: defaultdict(dict))

    for technique in TECHNIQUES:
        results_dir = os.path.join(experiments_dir, technique, "results")
        if not os.path.isdir(results_dir):
            continue

        # Walk: results/{model}/{dataset}/*.json
        for model_name in sorted(os.listdir(results_dir)):
            model_dir = os.path.join(results_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            for dataset_name in sorted(os.listdir(model_dir)):
                dataset_dir = os.path.join(model_dir, dataset_name)
                if not os.path.isdir(dataset_dir):
                    continue

                json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
                chosen = pick_latest_file(json_files)
                if chosen is None:
                    continue

                try:
                    with open(chosen) as f:
                        results = json.load(f)
                except Exception as e:
                    print(f"  Warning: could not read {chosen}: {e}")
                    continue

                # Handle both list format and dict-with-results format
                if isinstance(results, dict) and "results" in results:
                    results_list = results["results"]
                elif isinstance(results, list):
                    results_list = results
                else:
                    print(f"  Warning: unexpected format in {chosen}")
                    continue

                correct, total, acc = compute_accuracy(results_list)

                # Merge typo alias
                canonical_model = model_name
                if model_name == "HAIR_LIMO-v2":
                    canonical_model = "GAIR_LIMO-v2"

                data[dataset_name][technique][canonical_model] = acc
                # print(f"  {technique}/{canonical_model}/{dataset_name}: {correct}/{total} = {acc:.1f}%")

    return data


def shorten(name, mapping):
    return mapping.get(name, name)

# ── Color palette ─────────────────────────────────────────────────

# Professional color palette
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
]

# ── Plotting ─────────────────────────────────────────────────────────

def plot_dataset(dataset_name, technique_data, outdir):
    """
    Create one large figure for a dataset with one subplot per technique (bar chart).
    technique_data: dict[technique] -> dict[model] -> accuracy_pct
    """
    # Collect all models that appear in any technique for this dataset
    all_models = set()
    for td in technique_data.values():
        all_models.update(td.keys())
    all_models = sorted(all_models)

    if not all_models:
        print(f"  No models found for dataset {dataset_name}, skipping.")
        return

    # Filter techniques that have data
    techniques_with_data = [t for t in TECHNIQUES if t in technique_data and technique_data[t]]
    n_techniques = len(techniques_with_data)
    if n_techniques == 0:
        print(f"  No techniques with data for dataset {dataset_name}, skipping.")
        return

    # Assign consistent colors to models
    model_colors = {}
    for i, model in enumerate(all_models):
        model_colors[model] = PALETTE[i % len(PALETTE)]

    # Layout: aim for roughly 3-4 columns
    ncols = min(4, n_techniques)
    nrows = (n_techniques + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))

    # Handle single-row/col case
    if n_techniques == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    fig.suptitle(f"Model Accuracy — {dataset_label}", fontsize=18, fontweight='bold', y=0.98)

    for idx, technique in enumerate(techniques_with_data):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        td = technique_data[technique]

        x = np.arange(len(all_models))
        bar_width = 0.65
        accuracies = [td.get(m, 0.0) for m in all_models]
        colors = [model_colors[m] for m in all_models]
        labels = [shorten(m, MODEL_SHORT_NAMES) for m in all_models]

        bars = ax.bar(x, accuracies, bar_width, color=colors, edgecolor='white', linewidth=0.5)

        # Annotate bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                        f"{acc:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(TECHNIQUE_LABELS.get(technique, technique), fontsize=13, fontweight='bold', pad=8)
        ax.set_xticks(x)
        # Rotate x labels slightly and align to the right to avoid overlap.
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for idx in range(n_techniques, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Leave extra bottom space for rotated/staggered x-axis labels
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"results_{dataset_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results across models and transformations")
    parser.add_argument("--experiments_dir", type=str, default=None,
                        help="Path to experiments/ directory (auto-detected if not set)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter to a specific dataset (safe name, e.g. HuggingFaceH4_aime_2024)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for plots (defaults to analysis/plots/)")
    args = parser.parse_args()

    # Resolve experiments directory
    if args.experiments_dir:
        experiments_dir = args.experiments_dir
    else:
        # Auto-detect: script is in analysis/, experiments/ is parent
        script_dir = os.path.dirname(os.path.abspath(__file__))
        experiments_dir = os.path.dirname(script_dir)

    if not os.path.isdir(experiments_dir):
        print(f"Error: experiments directory not found at {experiments_dir}")
        sys.exit(1)

    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.join(experiments_dir, "analysis", "plots")

    print(f"Scanning results in: {experiments_dir}")
    data = scan_results(experiments_dir)

    if not data:
        print("No results found. Check that results/ directories exist under technique folders.")
        sys.exit(1)

    # Filter to specific dataset if requested
    datasets = sorted(data.keys())
    if args.dataset:
        datasets = [d for d in datasets if args.dataset in d]
        if not datasets:
            print(f"No results found for dataset filter '{args.dataset}'")
            sys.exit(1)

    print(f"\nDatasets found: {datasets}")
    print(f"Output dir: {outdir}\n")

    for dataset_name in datasets:
        print(f"Plotting: {dataset_name}")
        plot_dataset(dataset_name, data[dataset_name], outdir)

    print("\nDone!")


if __name__ == "__main__":
    main()
