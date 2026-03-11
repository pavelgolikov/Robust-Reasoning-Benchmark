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
    "context_saturation":           "Context Saturation",
}

# Canonical ordering of techniques for by-model plots
TECHNIQUE_ORDER = [
    "not_not", "opposites", "wrappers",
    "interleaved_context_line", "interleaved_context_word", "interleaved_context_symbol",
    "context_saturation",
    "sentence_reversal", "word_reversal", "split_reversal",
    "rail_fence",
]

# Shorten model names for x-axis
MODEL_SHORT_NAMES = {
    "GAIR_LIMO-v2":                                     "LIMO-v2\n(32B)",
    "tiiuae_Falcon-H1R-7B":                             "Falcon-H1R\n(7B)",
    "openai_gpt-oss-120b":                              "GPT-OSS\n(120B)",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-70B":        "DSR1-Llama\n(70B)",
    "Qwen_Qwen3.5-35B-A3B":                             "Qwen3.5-35B",
    "Qwen_Qwen3-30B-A3B-Thinking-2507":                 "Qwen3-30B",
    "gemini-3.1-pro-preview":                           "Gemini 3.1\nPro",
    "gemini-2.5-flash":                                 "Gemini 2.5\nFlash",
    "claude-opus-4-6":                                  "Claude Opus\n4-6",
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
    # Filter out files that are known to be metadata/input files
    actual_results = [f for f in json_files if not (os.path.basename(f).startswith("jobs_") or 
                                                    os.path.basename(f).startswith("tracking_") or
                                                    os.path.basename(f).startswith("batch_"))]
    if not actual_results:
        return None
    # Files have timestamps like _20260303_223355.json — sort lexicographically
    return sorted(actual_results)[-1]


def scan_results(experiments_dir, aggregate=False):
    """
    Scan all technique directories under experiments_dir.
    Returns:
    data[dataset][technique][model] = {
        'accuracy': accuracy_pct,
        'n_samples': n_samples
    }
    """
    data = defaultdict(lambda: defaultdict(dict))

    # Auto-discover techniques: any directory that has a 'results' subdirectory
    discovered_techniques = []
    for d in sorted(os.listdir(experiments_dir)):
        if os.path.isdir(os.path.join(experiments_dir, d, "results")):
            discovered_techniques.append(d)

    for technique in discovered_techniques:
        results_dir = os.path.join(experiments_dir, technique, "results")
        
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
                
                if aggregate:
                    # Collect all valid result files
                    target_files = [f for f in json_files if not (os.path.basename(f).startswith("jobs_") or 
                                                                    os.path.basename(f).startswith("tracking_") or
                                                                    os.path.basename(f).startswith("batch_"))]
                else:
                    chosen = pick_latest_file(json_files)
                    target_files = [chosen] if chosen else []
                
                if not target_files:
                    continue

                all_results = []
                for fpath in target_files:
                    try:
                        with open(fpath) as f:
                            content = json.load(f)
                        
                        # Handle both list format and dict-with-results format
                        if isinstance(content, dict) and "results" in content:
                            all_results.extend(content["results"])
                        elif isinstance(content, list):
                            all_results.extend(content)
                        elif isinstance(content, dict):
                            # Some might be mappings from ID to result
                            all_results.extend(list(content.values()))
                    except Exception as e:
                        print(f"  Warning: could not read {fpath}: {e}")
                        continue
                
                if not all_results:
                    continue

                correct, total, acc = compute_accuracy(all_results)
                
                # Calculate avg n_samples per problem
                unique_ids = set(r.get('id') for r in all_results if r.get('id') is not None)
                n_samples = total / len(unique_ids) if unique_ids else 0

                # Merge typo alias
                canonical_model = model_name
                if model_name == "HAIR_LIMO-v2":
                    canonical_model = "GAIR_LIMO-v2"

                data[dataset_name][technique][canonical_model] = {
                    'accuracy': acc,
                    'n_samples': n_samples
                }

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

def plot_dataset(dataset_name, technique_data, outdir, aggregate=False):
    """
    Create one large figure for a dataset with one subplot per technique (bar chart).
    technique_data: dict[technique] -> dict[model] -> {accuracy, n_samples}
    """
    # Collect all models that appear in any technique for this dataset
    all_models_global = set()
    for td in technique_data.values():
        all_models_global.update(td.keys())
    all_models_global = sorted(all_models_global)

    if not all_models_global:
        print(f"  No models found for dataset {dataset_name}, skipping.")
        return

    # Filter techniques that have data
    techniques_with_data = sorted(technique_data.keys(), key=lambda t: (t != 'baseline', t))
    
    n_techniques = len(techniques_with_data)
    if n_techniques == 0:
        print(f"  No techniques with data for dataset {dataset_name}, skipping.")
        return

    # Assign consistent colors to all models globally for consistency across subplots
    model_colors = {}
    for i, model in enumerate(all_models_global):
        model_colors[model] = PALETTE[i % len(PALETTE)]

    # Layout: aim for roughly 3-4 columns
    ncols = min(4, n_techniques)
    nrows = (n_techniques + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))

    # Handle single-row/col case
    if n_techniques == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    fig.suptitle(f"Model Accuracy — {dataset_label}", fontsize=20, fontweight='bold', y=0.98)

    for idx, technique in enumerate(techniques_with_data):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        td = technique_data[technique]
        
        # "Don't have bars where we don't have data"
        # Only models that are actually present in this technique
        subplot_models = [m for m in all_models_global if m in td]
        
        if not subplot_models:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(TECHNIQUE_LABELS.get(technique, technique), fontsize=13, fontweight='bold', pad=8)
            continue

        x = np.arange(len(subplot_models))
        bar_width = 0.65
        accuracies = [td[m]['accuracy'] for m in subplot_models]
        colors = [model_colors[m] for m in subplot_models]
        
        labels = []
        for m in subplot_models:
            short = shorten(m, MODEL_SHORT_NAMES)
            if aggregate:
                n = td[m]['n_samples']
                n_str = str(int(n)) if n == int(n) else f"{n:.1f}"
                labels.append(f"{short}\n(n={n_str})")
            else:
                labels.append(short)

        bars = ax.bar(x, accuracies, bar_width, color=colors, edgecolor='white', linewidth=0.5)

        # Annotate bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                    f"{acc:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

        title = TECHNIQUE_LABELS.get(technique, technique)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for idx in range(n_techniques, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"results_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def plot_by_model(dataset_name, technique_data, outdir, aggregate=False, per_model_pdfs=False):
    """
    Create a figure with one subplot per model, x-axis = transforms, y-axis = accuracy.
    Also optionally saves separate PDF files per model.
    technique_data: dict[technique] -> dict[model] -> {accuracy, n_samples}
    """
    # Pivot: model -> technique -> accuracy
    all_models = set()
    for td in technique_data.values():
        all_models.update(td.keys())

    if not all_models:
        print(f"  No models found for dataset {dataset_name}, skipping by-model plot.")
        return

    # Sort models by average accuracy across transforms (strongest first)
    def _avg_accuracy(model):
        accs = [technique_data[t][model]['accuracy']
                for t in technique_data if model in technique_data[t]]
        return sum(accs) / len(accs) if accs else 0

    all_models = sorted(all_models, key=_avg_accuracy, reverse=True)

    # Use canonical ordering, only include techniques that have data
    available_techniques = set(technique_data.keys())
    ordered_techniques = [t for t in TECHNIQUE_ORDER if t in available_techniques]
    # Append any techniques not in TECHNIQUE_ORDER
    for t in sorted(available_techniques):
        if t not in ordered_techniques:
            ordered_techniques.append(t)

    if not ordered_techniques:
        print(f"  No techniques with data for by-model plot, skipping.")
        return

    n_techniques = len(ordered_techniques)
    technique_labels = [TECHNIQUE_LABELS.get(t, t) for t in ordered_techniques]

    # Assign consistent colors per technique
    technique_colors = {}
    for i, t in enumerate(ordered_techniques):
        technique_colors[t] = PALETTE[i % len(PALETTE)]

    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)

    # ── Helper to plot a single model on an axis ──
    def _plot_model_on_ax(ax, model_name):
        accuracies = []
        bar_colors = []
        for t in ordered_techniques:
            td = technique_data.get(t, {})
            if model_name in td:
                accuracies.append(td[model_name]['accuracy'])
            else:
                accuracies.append(0)
            bar_colors.append(technique_colors[t])

        x = np.arange(n_techniques)
        bar_width = 0.65
        bars = ax.bar(x, accuracies, bar_width, color=bar_colors, edgecolor='white', linewidth=0.5)

        # Annotate bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                        f"{acc:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

        model_label = shorten(model_name, MODEL_SHORT_NAMES).replace('\n', ' ')
        ax.set_title(model_label, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(technique_labels, fontsize=8, rotation=45, ha='right')
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Combined summary figure ──
    n_models = len(all_models)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    fig.suptitle(f"Accuracy by Transform — {dataset_label}", fontsize=20, fontweight='bold', y=0.98)

    for idx, model_name in enumerate(all_models):
        row, col = divmod(idx, ncols)
        _plot_model_on_ax(axes[row, col], model_name)

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"by_model_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved summary: {out_path}")

    # ── Per-model PDFs ──
    if per_model_pdfs:
        pdf_dir = os.path.join(outdir, "per_model")
        os.makedirs(pdf_dir, exist_ok=True)
        for model_name in all_models:
            fig_m, ax_m = plt.subplots(figsize=(max(8, n_techniques * 0.8), 5.5))
            _plot_model_on_ax(ax_m, model_name)
            model_label = shorten(model_name, MODEL_SHORT_NAMES).replace('\n', ' ')
            fig_m.suptitle(f"{model_label} — {dataset_label}", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            safe_model = model_name.replace('/', '_').replace(' ', '_')
            pdf_path = os.path.join(pdf_dir, f"{safe_model}_{dataset_name}.pdf")
            fig_m.savefig(pdf_path, bbox_inches='tight', facecolor='white')
            plt.close(fig_m)
            print(f"  Saved PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results across models and transformations")
    parser.add_argument("--experiments_dir", type=str, default=None,
                        help="Path to experiments/ directory (auto-detected if not set)")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4_aime_2024",
                        help="Filter to a specific dataset (safe name, e.g. HuggingFaceH4_aime_2024)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for plots (defaults to analysis/plots/)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate results across all JSON files for each model/technique (not just the latest)")
    parser.add_argument("--by_model", action="store_true",
                        help="Plot by model (x-axis = transforms, one subplot per model)")
    parser.add_argument("--per_model_pdfs", action="store_true",
                        help="When --by_model is set, also save separate PDF files per model")
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
    data = scan_results(experiments_dir, aggregate=args.aggregate)

    if not data:
        print("No results found. Check that results/ directories exist under technique folders.")
        sys.exit(1)

    # Filter to specific dataset if requested
    datasets = sorted(data.keys())
    # Exclude non-paper results
    datasets = [d for d in datasets if d != 'non_paper']
    if args.dataset:
        datasets = [d for d in datasets if args.dataset in d]
        if not datasets:
            print(f"No results found for dataset filter '{args.dataset}'")
            sys.exit(1)

    print(f"\nDatasets found: {datasets}")
    print(f"Output dir: {outdir}\n")

    for dataset_name in datasets:
        print(f"Plotting: {dataset_name}")
        if args.by_model:
            plot_by_model(dataset_name, data[dataset_name], outdir,
                          aggregate=args.aggregate, per_model_pdfs=args.per_model_pdfs)
        else:
            plot_dataset(dataset_name, data[dataset_name], outdir, aggregate=args.aggregate)

    print("\nDone!")


if __name__ == "__main__":
    main()
