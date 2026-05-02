#!/usr/bin/env python3
"""
Optimized Visualization Script for Linguistic Traps Evaluation.
Sources data from pre-generated summary JSONs in:
- experiments/analysis/output_length/results/
- experiments/analysis/prompt_reconstruction/results/
"""

import argparse
import json
import os
import glob
import sys
import re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as patheffects
import matplotlib.patches as patches

# ── Configuration ────────────────────────────────────────────────────

TECHNIQUE_LABELS = {
    "baseline":                     "Baseline",
    "opposites":                    "Opposites",
    "not_not":                      "Not-Not",
    "wrappers":                     "Wrappers",
    "split_reversal":               "Split-Rev",
    "word_reversal":                "Word-Rev",
    "sentence_reversal":            "Sentence-Rev",
    "rail_fence":                   "Rail Fence",
    "interleaved_context_line":     "Interleave-L",
    "interleaved_context_word":     "Interleave-W",
    "interleaved_context_symbol":   "Interleave-S",
    "rectangle_perimeter":          "Rectangle",
    "snake_vertical":               "Snake-V",
    "snake_horizontal":             "Snake-H",
}

TECHNIQUE_ORDER = [
    "baseline",
    "not_not", "opposites", "wrappers",
    "interleaved_context_line", "interleaved_context_word", "interleaved_context_symbol",
    "sentence_reversal", "word_reversal", "split_reversal",
    "rail_fence",
    "rectangle_perimeter", "snake_vertical", "snake_horizontal",
]

MODEL_SHORT_NAMES = {
    "Qwen_Qwen3-30B-A3B-Thinking-2507":                 "Qwen3-30B-A3B",
    "nvidia_OpenReasoning-Nemotron-7B":                 "Nemotron-7B",
    "nvidia_OpenReasoning-Nemotron-32B":                "Nemotron-32B",
    "openai_gpt-oss-120b":                              "GPT-OSS-120B",
    "gpt-5.4":                                          "GPT-5.4",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-70B":        "DSR1-Llama-70B",
    "gemini-3.1-pro-preview":                           "Gemini 3.1 Pro",
    "claude-opus-4-6":                                  "Opus 4.6",
}

DATASET_SHORT_NAMES = {
    "HuggingFaceH4_aime_2024":  "AIME 2024",
    "MathArena_aime_2025":      "AIME 2025",
    "MATH_500":                 "MATH 500",
    "MathArena_hmmt_feb_2025":  "HMMT Feb 2025",
}

CATEGORIES = {
    "Syntactic": ["split_reversal", "word_reversal", "sentence_reversal"],
    "Semantic": ["not_not", "opposites", "wrappers"],
    "Visual": ["rail_fence"],
    "Contextual": ["interleaved_context_line", "interleaved_context_symbol", "interleaved_context_word"],
}

CATEGORY_NAMES = list(CATEGORIES.keys())

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", 
    "#937860", "#DA8BC3", "#64B5CD", "#CCB974", "#636363", 
    "#764978", "#006400", "#8B0000"
]

# ── Helpers ──────────────────────────────────────────────────────────

def shorten(name, mapping):
    return mapping.get(name, name)

def format_length_label(value):
    return f"{value/1000:.0f}"

# ── Fast Data Loading ───────────────────────────────────────────────

def get_latest_json(target_dir):
    json_files = glob.glob(os.path.join(target_dir, "*.json"))
    valid_files = [f for f in json_files if not f.endswith("_raw.json") and "_summary_" not in f and "prompt_recovery" not in f]
    if not valid_files:
        return None
    return max(valid_files, key=os.path.getmtime)

def load_metrics_data(experiments_dir, safe_dataset):
    """Load accuracy, failure, and length data directly from transformation result JSONs."""
    data = defaultdict(lambda: defaultdict(dict))
    
    for technique in TECHNIQUE_ORDER:
        tech_dir = os.path.join(experiments_dir, technique, "results")
        if not os.path.isdir(tech_dir):
            continue
            
        for model in os.listdir(tech_dir):
            if model not in MODEL_SHORT_NAMES:
                continue
                
            model_dataset_dir = os.path.join(tech_dir, model, safe_dataset)
            if not os.path.isdir(model_dataset_dir):
                continue
                
            perturb_dir = os.path.join(model_dataset_dir, "perturb")
            if os.path.isdir(perturb_dir):
                target_dir = perturb_dir
            else:
                target_dir = model_dataset_dir
                
            latest_file = get_latest_json(target_dir)
            if not latest_file:
                continue
                
            try:
                with open(latest_file, 'r') as f:
                    content = json.load(f)
                    
                acc = 0.0
                fail_rate = 0.0
                avg_length = 0.0
                
                # Support dictionary wrapping or list structure
                if isinstance(content, list) and len(content) > 0:
                    summary = content[-1].get("summary", {}) if isinstance(content[-1], dict) else {}
                elif isinstance(content, dict):
                    summary = content.get("summary", {})
                else:
                    summary = {}
                    
                # Define results for fallback computation
                results = content if isinstance(content, list) else content.get("results", [])
                    
                avg_length = summary.get("avg_output_tokens", 0.0)
                
                # If avg_output_tokens is missing, compute it from results
                if avg_length == 0.0 and results:
                    token_counts = [r.get("output_tokens", 0) for r in results if isinstance(r, dict) and r.get("id") is not None]
                    if token_counts:
                        avg_length = sum(token_counts) / len(token_counts)
                
                if summary:
                    # Parse from summary block
                    acc = summary.get("accuracy", 0.0)
                    if isinstance(acc, float) and acc <= 1.0 and acc > 0:
                        acc *= 100.0
                    
                    refusals = summary.get("refusals", 0)
                    total = summary.get("total", 0)
                    attempted = total - refusals
                    
                    if attempted > 0:
                        acc_attempted = summary.get("accuracy_attempted", None)
                        if acc_attempted is None:
                            correct = summary.get("correct", 0)
                            acc_attempted = (correct / attempted)
                        
                        if isinstance(acc_attempted, (float, int)) and acc_attempted <= 1.0 and acc_attempted > 0:
                            acc_attempted *= 100.0
                        elif acc_attempted == 0:
                            acc_attempted = 0.0
                    else:
                        acc_attempted = None
                        
                    n_failures = refusals + summary.get("failures", 0)
                    fail_rate = (n_failures / total * 100.0) if total > 0 else 0.0
                else:
                    acc = 0.0
                    acc_attempted = None
                    fail_rate = 0.0
                    
                # Dynamically compute from raw results array if summary misses fields
                if "accuracy" not in summary:
                    total_computed = 0
                    correct = 0
                    n_refusals_computed = 0
                    n_failures_computed = 0
                    for r in results:
                        if isinstance(r, dict) and r.get("id") is not None:
                            total_computed += 1
                            if r.get("correct", False):
                                correct += 1
                            if r.get("refusal") is True:
                                n_refusals_computed += 1
                            elif (not r.get("correct") and r.get("extracted") is None):
                                n_failures_computed += 1
                                
                    acc = (correct / total_computed * 100.0) if total_computed > 0 else 0.0
                    attempted_computed = total_computed - n_refusals_computed
                    acc_attempted = (correct / attempted_computed * 100.0) if attempted_computed > 0 else None
                    fail_rate = ((n_refusals_computed + n_failures_computed) / total_computed * 100.0) if total_computed > 0 else 0.0
                    total = total_computed
                    attempted = attempted_computed
                    
                data[technique][model] = {
                    'accuracy': acc,
                    'accuracy_attempted': acc_attempted,
                    'failure_rate': fail_rate,
                    'length': avg_length,
                    'total': total,
                    'attempted': attempted
                }
            except ValueError as ve:
                print(ve)
                import sys
                sys.exit(1)
            except Exception:
                continue
                
    return data



def load_recovery_and_conditional(experiments_dir, safe_dataset):
    """Load data from analysis/prompt_reconstruction/results/ reports."""
    report_base = os.path.join(experiments_dir, "analysis", "prompt_reconstruction", "results")
    rec_data = defaultdict(dict)
    cond_data = defaultdict(lambda: defaultdict(dict))
    
    if not os.path.isdir(report_base):
        return rec_data, cond_data

    for model_name in os.listdir(report_base):
        if model_name not in MODEL_SHORT_NAMES:
            continue
            
        model_dataset_dir = os.path.join(report_base, model_name, safe_dataset)
        if not os.path.isdir(model_dataset_dir):
            continue
            
        for f in os.listdir(model_dataset_dir):
            if f.endswith(".json") and "prompt_recovery" in f:
                technique = None
                for t in TECHNIQUE_ORDER:
                    if f.startswith(t + "_prompt_recovery"):
                        technique = t
                        break
                if not technique: continue
                
                fpath = os.path.join(model_dataset_dir, f)
                try:
                    with open(fpath) as j:
                        report = json.load(j)
                    
                    # Recovery Rate
                    sem_acc = report.get('semantic_accuracy', 0) * 100
                    rec_data[technique][model_name] = {
                        'recovery_rate': sem_acc,
                        'n_samples': report.get('total_samples', 0)
                    }
                    
                    # Conditional Accuracy
                    orig_correct = report.get('original_correct', 0)
                    sem_correct = report.get('semantic_correct', 0)
                    cond_acc = 100.0 * orig_correct / sem_correct if sem_correct > 0 else 0.0
                    cond_data[model_name][technique] = {
                        'conditional_accuracy': cond_acc,
                        'n_recovered': sem_correct,
                        'n_total': report.get('total_samples', 0),
                        'n_solved': orig_correct
                    }
                except Exception:
                    continue
    return rec_data, cond_data

# ── Plotting ─────────────────────────────────────────────────────────

def plot_by_model(dataset_name, technique_data, outdir, metric='accuracy', failures_on_top=False):
    all_models = set()
    for td in technique_data.values(): all_models.update(td.keys())
    if not all_models: return

    def _avg_accuracy(model):
        accs = [technique_data[t][model]['accuracy'] for t in technique_data if model in technique_data[t]]
        return sum(accs) / len(accs) if accs else 0
    all_models = sorted(all_models, key=_avg_accuracy, reverse=True)

    ordered_techniques = TECHNIQUE_ORDER.copy()
    available_techniques = set(technique_data.keys())
    for t in sorted(available_techniques):
        if t not in ordered_techniques: ordered_techniques.append(t)

    n_techniques = len(ordered_techniques)
    technique_labels = [TECHNIQUE_LABELS.get(t, t) for t in ordered_techniques]
    technique_colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(ordered_techniques)}

    def _plot_model_on_ax(ax, model_name):
        accuracies, fail_rates, bar_colors, is_missing = [], [], [], []
        for t in ordered_techniques:
            td = technique_data.get(t, {})
            if model_name in td:
                accuracies.append(td[model_name][metric])
                fail_rates.append(td[model_name].get('failure_rate', 0))
                is_missing.append(False)
            else:
                accuracies.append(0); fail_rates.append(0); is_missing.append(True)
            bar_colors.append(technique_colors[t])

        x = np.arange(n_techniques)
        bars = ax.bar(x, accuracies, 0.65, color=bar_colors, edgecolor='black', linewidth=0.5)

        if failures_on_top and metric == 'accuracy':
            ax.bar(x, fail_rates, 0.65, bottom=accuracies, color=bar_colors, 
                   edgecolor='black', linewidth=0.5, alpha=0.4, hatch='///')

        for bar, acc, missing, t in zip(bars, accuracies, is_missing, ordered_techniques):
            if missing: text = "N/A"
            elif metric == 'length':
                text = format_length_label(acc)
                acc_val = technique_data[t][model_name]['accuracy']
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                        f"{acc_val:.0f}", ha='center', va='center', fontsize=10, color='white', fontweight='bold',
                        path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
            else: text = f"{acc:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, text, ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_title(shorten(model_name, MODEL_SHORT_NAMES).replace('\n', ' '), fontsize=15, fontweight='bold', pad=10)
        ax.set_xticks(x); ax.set_xticklabels(technique_labels, fontsize=12, rotation=45, ha='right')
        ax.set_ylabel("Length (tokens)" if metric == 'length' else "Accuracy (%)", fontsize=12)
        if metric != 'length':
            ax.set_ylim(0, 115); ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ncols, n_models = 2, len(all_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)
    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    # fig.suptitle(f"{('Length' if metric=='length' else 'Accuracy')} by Transform — {dataset_label}", fontsize=22, fontweight='bold', y=0.98)

    for idx, model_name in enumerate(all_models):
        row, col = divmod(idx, ncols); _plot_model_on_ax(axes[row, col], model_name)
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols); axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{metric}_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_recovery(rec_data, dataset_name, outdir, accuracy_data=None, accuracy_overlay=False):
    all_models = set()
    for td in rec_data.values(): all_models.update(td.keys())
    if not all_models: return

    def _avg_accuracy(model):
        if not accuracy_data: return 0
        accs = [accuracy_data[t][model]['accuracy'] for t in accuracy_data if model in accuracy_data[t]]
        return sum(accs) / len(accs) if accs else 0
    all_models = sorted(all_models, key=_avg_accuracy, reverse=True)

    ordered_techniques = TECHNIQUE_ORDER.copy()
    for t in sorted(rec_data.keys()):
        if t not in ordered_techniques: ordered_techniques.append(t)
    
    technique_colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(ordered_techniques)}

    def _plot_model_on_ax(ax, model_name):
        rates, bar_colors, is_missing = [], [], []
        for t in ordered_techniques:
            td = rec_data.get(t, {})
            if model_name in td: rates.append(td[model_name]['recovery_rate']); is_missing.append(False)
            elif t in ["baseline", "context_saturation"]: rates.append(100); is_missing.append(False)
            else: rates.append(0); is_missing.append(True)
            bar_colors.append(technique_colors[t])

        x = np.arange(len(ordered_techniques))
        bars = ax.bar(x, rates, 0.65, color=bar_colors, edgecolor='black', linewidth=0.5)

        if accuracy_overlay and accuracy_data:
            overlay = [accuracy_data.get(t, {}).get(model_name, {}).get('accuracy', 0) for t in ordered_techniques]
            ax.bar(x, overlay, 0.65, color='none', edgecolor='black', linewidth=0.5, hatch='//', alpha=0.6)

        for bar, rate, missing in zip(bars, rates, is_missing):
            text = "N/A" if missing else f"{rate:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, text, ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_title(shorten(model_name, MODEL_SHORT_NAMES).replace('\n', ' '), fontsize=15, fontweight='bold', pad=10)
        ax.set_xticks(x); ax.set_xticklabels([TECHNIQUE_LABELS.get(t, t) for t in ordered_techniques], fontsize=12, rotation=45, ha='right')
        ax.set_ylabel("Recovery Rate (%)", fontsize=12); ax.set_ylim(0, 115)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ncols, n_models = 2, len(all_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)
    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    # fig.suptitle(f"Prompt Recovery Rate by Model — {dataset_label}", fontsize=22, fontweight='bold', y=0.98)

    for idx, model_name in enumerate(all_models):
        row, col = divmod(idx, ncols); _plot_model_on_ax(axes[row, col], model_name)
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols); axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"prompt_recovery_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_single_metric(dataset_name, technique_data, outdir, exclude_refusals=False):
    all_models = set()
    for td in technique_data.values(): all_models.update(td.keys())
    if not all_models: return

    def _avg_acc(model):
        accs = [technique_data[t][model]['accuracy'] for t in technique_data if model in technique_data[t]]
        return sum(accs) / len(accs) if accs else 0
    all_models = sorted(all_models, key=_avg_acc, reverse=True)

    model_deltas = {}
    model_attempt_rates = {}
    metric_key = 'accuracy_attempted' if exclude_refusals else 'accuracy'
    
    for model in all_models:
        baseline_acc = technique_data.get('baseline', {}).get(model, {}).get(metric_key)
        if baseline_acc is None: continue
        
        deltas = []
        total_s = 0
        total_a = 0
        for t in technique_data:
            if model not in technique_data[t]: continue
            
            # Aggregate attempt rates for all techniques including baseline
            total_s += technique_data[t][model].get('total', 0)
            total_a += technique_data[t][model].get('attempted', 0)
            
            if t == 'baseline': continue
            
            acc = technique_data[t][model].get(metric_key)
            if acc is not None:
                deltas.append(baseline_acc - acc)
                
        if deltas:
            model_deltas[model] = sum(deltas) / len(deltas)
            model_attempt_rates[model] = total_a / total_s if total_s > 0 else 0.0

    if not model_deltas:
        print("No models have valid data for average accuracy drop.")
        return

    plot_models = sorted(model_deltas.keys(), key=lambda m: model_deltas[m])
    values = [model_deltas[m] for m in plot_models]
    colors = [PALETTE[all_models.index(m) % len(PALETTE)] for m in plot_models]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(np.arange(len(plot_models)), values, 0.65, color=colors, edgecolor='black', linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, values)):
        model = plot_models[i]
        # Standard accuracy drop label - always place 'above' the bar end
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", \
            ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        # Add 'attempted' highlight for Claude above the bar
        if exclude_refusals and "claude" in model.lower():
            rate = model_attempt_rates.get(model, 1.0) * 100.0
            # Position it higher than the drop value
            ax.text(bar.get_x(), bar.get_height() + 6.0, 
                    f"{rate:.1f}% attempted", 
                    ha='left', va='bottom', fontsize=15, fontweight='bold', 
                    color='darkred', rotation=70)

    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    # ax.set_title(f"Average Accuracy Drop — {dataset_label}", fontsize=22, fontweight='bold', pad=20)
    ax.set_xticks(np.arange(len(plot_models)))
    
    labels = []
    for m in plot_models:
        name = shorten(m, MODEL_SHORT_NAMES).replace('\n', ' ')
        if exclude_refusals:
            rate = model_attempt_rates.get(m, 1.0) * 100.0
            # Use the internal ID 'm' to identify Claude models
            if "claude" in m.lower():
                labels.append(f"{name}")
            else:
                labels.append(name)
        else:
            labels.append(name)
            
    ax.set_xticklabels(labels, fontsize=18, rotation=45, ha='right')
    ax.set_ylabel("Avg Accuracy Drop (%)", fontsize=20)
    ax.set_ylim(min(values + [0]) * 1.1, max(values + [0]) * 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    suffix = "_no_refusals" if exclude_refusals else ""
    out_path = os.path.join(outdir, f"average_accuracy_drop{suffix}_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_radar_charts(dataset_name, technique_data, outdir):
    all_models = set()
    for td in technique_data.values(): all_models.update(td.keys())
    if not all_models: return
    model_category_acc = defaultdict(dict)
    for model in all_models:
        for cat_name, techniques in CATEGORIES.items():
            accs = [technique_data[t][model]['accuracy'] for t in techniques if t in technique_data and model in technique_data[t]]
            model_category_acc[model][cat_name] = sum(accs) / len(accs) if accs else 0.0

    def _avg_acc(model):
        accs = [technique_data[t][model]['accuracy'] for t in technique_data if model in technique_data[t]]
        return sum(accs) / len(accs) if accs else 0
    models_to_plot = sorted(list(all_models), key=_avg_acc, reverse=True)

    ncols, n_models = 4, len(models_to_plot)
    nrows = (n_models + ncols - 1) // ncols
    fig = plt.figure(figsize=(5.0 * ncols, 4.5 * nrows))
    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    # fig.suptitle(f"Performance by Category — {dataset_label}", fontsize=22, fontweight='bold', y=0.98)

    angles = np.linspace(0, 2 * np.pi, len(CATEGORY_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    for i, model in enumerate(models_to_plot):
        ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
        values = [model_category_acc[model][cat] for cat in CATEGORY_NAMES]
        values += values[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, values, color=color, linewidth=2); ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_ylim(0, 100); ax.set_rlabel_position(0)
        ax.set_rgrids([20, 40, 60, 80], labels=["20", "40", "60", "80"], fontsize=14, color='gray')
        
        # Manually place category labels inside the circles
        for angle, label in zip(angles[:-1], CATEGORY_NAMES):
            ha, va = 'center', 'center'
            r_label = 95 # Inside the 100 limit
            
            if label == "Syntactic": 
                va = 'top' # Shift slightly below center of the angle
            elif label == "Visual": 
                va = 'bottom' # Shift slightly above center
            elif label == "Semantic": 
                ha, va = 'right', 'bottom' # Inside and slightly up
                angle -= 0.10
            elif label == "Contextual": 
                ha, va = 'left', 'top' # Inside and shifted down towards bottom
                angle -= 0.10 # Shift counter-clockwise (towards bottom)
            
            ax.text(angle, r_label, label, ha=ha, va=va, fontsize=16, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        ax.set_title(shorten(model, MODEL_SHORT_NAMES).replace('\n', ' '), size=15, fontweight='bold', pad=20)
        ax.set_xticklabels([]) # Hide default xticks
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(outdir, f"radar_categories_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_conditional_accuracy(dataset_name, cond_data, technique_data, outdir):
    all_models = sorted(cond_data.keys(), key=lambda m: sum(technique_data[t][m]['accuracy'] for t in technique_data if m in technique_data[t])/len([t for t in technique_data if m in technique_data[t]]) if any(m in technique_data[t] for t in technique_data) else 0, reverse=True)
    ncols, n_models = 2, len(all_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)
    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    # fig.suptitle(f"Accuracy Given Recovery — {dataset_label}", fontsize=22, fontweight='bold', y=0.98)

    tech_colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(TECHNIQUE_ORDER)}

    for idx, model_name in enumerate(all_models):
        row, col = divmod(idx, ncols); ax = axes[row, col]
        m_techs = cond_data[model_name]
        for t in ["baseline", "context_saturation"]:
            if t in technique_data and model_name in technique_data[t]:
                acc = technique_data[t][model_name]['accuracy']
                m_techs[t] = {'conditional_accuracy': acc, 'solve_pct': acc}
        
        plot_techs = [t for t in TECHNIQUE_ORDER if t in m_techs]
        x = np.arange(len(plot_techs))
        accs = [m_techs[t]['conditional_accuracy'] for t in plot_techs]
        bars = ax.bar(x, accs, 0.65, color=[tech_colors[t] for t in plot_techs], edgecolor='black', linewidth=0.5)

        for bar, val, tech in zip(bars, accs, plot_techs):
            pct = m_techs[tech].get('solve_pct', m_techs[tech].get('n_solved',0)*100/m_techs[tech].get('n_total',1))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{val:.1f}\n({pct:.1f})", ha='center', va='bottom', fontsize=8, fontweight='bold')

        # ax.set_title(shorten(model_name, MODEL_SHORT_NAMES).replace('\n', ' '), fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(x); ax.set_xticklabels([shorten(t, TECHNIQUE_LABELS) for t in plot_techs], fontsize=10, rotation=45, ha='right')
        ax.set_ylabel("Cond. Acc (%)", fontsize=11); ax.set_ylim(0, 125)
        ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False)

    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols); axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(outdir, f"conditional_accuracy_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_global_conditional_accuracy(dataset_name, cond_data, technique_data, outdir):
    exclude_techs = ["baseline", "context_saturation"]
    plot_data = []
    for model, techs in cond_data.items():
        sum_s = sum(metrics.get('n_solved', 0) for t, metrics in techs.items() if t not in exclude_techs)
        sum_r = sum(metrics.get('n_recovered', 0) for t, metrics in techs.items() if t not in exclude_techs)
        base = technique_data.get("baseline", {}).get(model, {}).get('accuracy', 0)
        plot_data.append({'model': model, 'base': base, 'g_cond': 100.0 * sum_s / sum_r if sum_r > 0 else 0.0})
    
    plot_data = sorted(plot_data, key=lambda x: x['g_cond'], reverse=True)
    models = [d['model'] for d in plot_data]
    x = np.arange(len(models)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width/2, [d['base'] for d in plot_data], width, label='Accuracy on\nOriginal', color='#4C72B0', edgecolor='black', linewidth=0.5)
    r2 = ax.bar(x + width/2, [d['g_cond'] for d in plot_data], width, label='Accuracy on\nRecovered', color='#DD8452', edgecolor='black', linewidth=0.5)
    
    # ax.set_title(f"Global Reasoning Stability — {shorten(dataset_name, DATASET_SHORT_NAMES)}", fontsize=22, fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=20)
    ax.set_xticks(x); ax.set_xticklabels([shorten(m, MODEL_SHORT_NAMES).replace('\n', ' ') for m in models], rotation=45, ha='right', fontsize=20)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.8); ax.set_ylim(0, 115); ax.grid(axis='y', alpha=0.3)
    
    for r in list(r1) + list(r2):
        ax.annotate(f'{r.get_height():.0f}', xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Highlight Open Source Models
    os_keywords = ["Nemotron", "DeepSeek", "Qwen", "gpt-oss"]
    os_indices = []
    for i, model in enumerate(models):
        if any(kw.lower() in model.lower() for kw in os_keywords):
            os_indices.append(i)
    
    if os_indices:
        x_start = min(os_indices) - 0.5
        x_end = max(os_indices) + 0.5
        y_max = max([d['base'] for i, d in enumerate(plot_data) if i in os_indices] + 
                    [d['g_cond'] for i, d in enumerate(plot_data) if i in os_indices])
        
        # Draw red bounding box around the tops of the bars
        box_y_min = 50
        box_y_max = y_max + 11
        rect = patches.Rectangle((x_start, box_y_min), x_end - x_start, box_y_max - box_y_min, 
                                 linewidth=4, edgecolor='red', facecolor='none', linestyle='-', zorder=5)
        ax.add_patch(rect)
        ax.text((x_start + x_end)/2 - 1.3, box_y_max + 1, "Open Weights Gap", color='red', 
                ha='center', va='bottom', fontsize=20, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(outdir, f"global_conditional_accuracy_{dataset_name}.pdf")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_compound(dataset_name, outdir, experiments_dir):
    compound_dir = os.path.join(experiments_dir, "compound", "results")
    baseline_dir = os.path.join(experiments_dir, "baseline", "results")
    
    if not os.path.isdir(compound_dir):
        raise RuntimeError(f"Compound directory not found: {compound_dir}")
        
    models_found = [d for d in os.listdir(compound_dir) if os.path.isdir(os.path.join(compound_dir, d))]
    
    if not models_found:
        raise RuntimeError(f"No models found in {compound_dir}")
        
    model_data = defaultdict(dict) # model -> {position: accuracy}
    
    for model in models_found:
        # Load baseline
        bl_model_dataset_dir = os.path.join(baseline_dir, model, dataset_name, "compound")
        if not os.path.isdir(bl_model_dataset_dir):
            raise RuntimeError(f"Baseline compound directory missing for {model}: {bl_model_dataset_dir}")
            
        bl_files = glob.glob(os.path.join(bl_model_dataset_dir, "*.json"))
        if not bl_files:
            raise RuntimeError(f"No baseline JSON found for {model} in {bl_model_dataset_dir}")
        if len(bl_files) > 1:
            raise RuntimeError(f"Multiple baseline JSONs found for {model} in {bl_model_dataset_dir}. Unsure which to use.")
            
        with open(bl_files[0]) as f:
            b_data = json.load(f)
            if isinstance(b_data, list):
                if len(b_data) > 0 and isinstance(b_data[-1], dict):
                    b_acc = b_data[-1].get("summary", {}).get("accuracy", 0.0)
                else:
                    b_acc = 0.0
            else:
                b_acc = b_data.get("summary", {}).get("accuracy", 0.0)
                
            if isinstance(b_acc, float) and b_acc <= 1.0 and b_acc > 0:
                b_acc *= 100.0
            model_data[model][1] = float(b_acc)
            
        # Load compound runs
        cp_model_dataset_dir = os.path.join(compound_dir, model, dataset_name)
        if not os.path.isdir(cp_model_dataset_dir):
            raise RuntimeError(f"Compound runs directory missing for {model}: {cp_model_dataset_dir}")
            
        cp_files = glob.glob(os.path.join(cp_model_dataset_dir, "*.json"))
        if not cp_files:
            raise RuntimeError(f"No compound JSONs found for {model} in {cp_model_dataset_dir}")
            
        for cpf in cp_files:
            with open(cpf) as f:
                c_data = json.load(f)
                
            if isinstance(c_data, list) and len(c_data) > 0:
                last_item = c_data[-1]
                first_item = c_data[0] if len(c_data) > 1 else c_data[0]
            else:
                last_item = c_data
                first_item = c_data
                
            summary = last_item.get("summary", {})
            
            correct = summary.get("correct", 0)
            total = summary.get("total", 0)
            cutoffs = summary.get("max_token_cutoffs", 0)
            
            if total > 0:
                c_acc = (correct + cutoffs) / float(total)
            else:
                c_acc = summary.get("accuracy", 0.0)
                
            if isinstance(c_acc, float) and c_acc <= 1.0 and c_acc > 0:
                c_acc *= 100.0
                
            orig = first_item.get("original", "")
            distractors = max(0, len(re.findall(r'Problem \d+:', orig)) - 1)
            if distractors == 0:
                prop_models = ["claude-opus-4-6", "gpt-5.4", "gemini-3.1-pro-preview"]
                if model in prop_models:
                    distractors = 3
            
            if distractors is None or distractors < 1:
                raise RuntimeError(f"Failed to determine valid distractor count in {cpf}")
                
            position = distractors + 1
            if position in model_data[model]:
                raise RuntimeError(f"Duplicate position {position} found for {model}. Files conflict.")
                
            model_data[model][position] = float(c_acc)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    
    target_models = [
        "gemini-3.1-pro-preview",
        "claude-opus-4-6",
        "gpt-5.4",
        "openai_gpt-oss-120b",
        "Qwen_Qwen3-30B-A3B-Thinking-2507",
        "nvidia_OpenReasoning-Nemotron-32B",
        "nvidia_OpenReasoning-Nemotron-7B",
        "deepseek-ai_DeepSeek-R1-Distill-Llama-70B"
    ]
    models_to_plot = [m for m in models_found if m in target_models]
    # Sort models by baseline accuracy
    models_to_plot = sorted(models_to_plot, key=lambda m: model_data[m].get(1, 0), reverse=True)
    
    # Swap claude-opus-4-6 and gpt-5.4
    try:
        idx_claude = models_to_plot.index("claude-opus-4-6")
        idx_gpt = models_to_plot.index("gpt-5.4")
        models_to_plot[idx_claude], models_to_plot[idx_gpt] = models_to_plot[idx_gpt], models_to_plot[idx_claude]
    except ValueError:
        pass
    
    x_ticks = []
    x_tick_labels = []
    model_centers = []
    model_labels = []
    
    current_x = 0
    
    for idx, model in enumerate(models_to_plot):
        positions = sorted(model_data[model].keys())
        accuracies = [model_data[model][p] for p in positions]
        
        current_x_base = current_x
        x_vals = []
        for p in positions:
            x_val = current_x_base + p
            x_vals.append(x_val)
            x_ticks.append(x_val)
            x_tick_labels.append(str(p))
            
        color = PALETTE[idx % len(PALETTE)]
        label_name = shorten(model, MODEL_SHORT_NAMES).replace('\n', ' ')
        
        ax.plot(x_vals, accuracies, marker='s', markersize=12, linewidth=4, 
                color=color, label=label_name)
                
        if x_vals:
            model_centers.append(sum(x_vals) / len(x_vals))
            model_labels.append(label_name)
                
        if len(accuracies) > 0:
            drop = accuracies[-1] - accuracies[0]
            ax.text(x_vals[-1] - 1.5, accuracies[-1] - 3.5, f"{drop:+.1f}%", 
                    color='red', fontsize=22, fontweight='bold', ha='left', va='center')
            
        # Ensure a uniform gap of 2 units between segments
        current_x = x_vals[-1] + 1  
                
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Place model labels explicitly underneath the sub-ticks
    for center, label in zip(model_centers, model_labels):
        ax.text(center, -0.12, label, transform=ax.get_xaxis_transform(), 
                ha='center', va='top', fontsize=22, rotation=45)
                
    ax.set_ylabel("Accuracy on Last Problem (%)", fontsize=22)
    # ax.set_xlabel("Number of problems asked per model", fontsize=18)
    
    ax.set_ylim(50, 105)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # Adjust margins to prevent text overflow (DSR1 percent on right, Y-label on top)
    fig.subplots_adjust(bottom=0.35, right=0.95, top=0.92)
    
    # Ensure the rightmost text (delta percentage) has enough space
    xmax = x_ticks[-1] + 1
    ax.set_xlim(left=0, right=xmax)
    
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"compound_{dataset_name}.pdf")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

def _segment_output_by_problems(output):
    """Split output into sections by first occurrence of each 'Problem X' marker."""
    pattern = re.compile(r"Problem\s*(\d+)")
    seen = {}
    for m in pattern.finditer(output):
        p = int(m.group(1))
        if p not in seen:
            seen[p] = m.start()
    if not seen:
        return {}
    sorted_markers = sorted(seen.items(), key=lambda x: x[1])
    sections = {}
    for i, (p, start) in enumerate(sorted_markers):
        end = sorted_markers[i + 1][1] if i + 1 < len(sorted_markers) else len(output)
        sections[p] = len(output[start:end])
    return sections


def plot_attention_effort(dataset_name, outdir, experiments_dir):
    """Stacked bar chart showing per-problem token effort ratios for compound experiments."""
    compound_dir = os.path.join(experiments_dir, "compound", "results")

    if not os.path.isdir(compound_dir):
        raise RuntimeError(f"Compound directory not found: {compound_dir}")

    ignore_models = ["ministral", "limo", "falcon", "gpt-5.4", "gemini", "claude", "deepseek"]

    models_found = [
        d for d in os.listdir(compound_dir)
        if os.path.isdir(os.path.join(compound_dir, d))
        and not any(m in d.lower() for m in ignore_models)
    ]

    if not models_found:
        raise RuntimeError(f"No eligible models found in {compound_dir}")

    # model -> {position: {problem_num: avg_ratio}}
    model_data = defaultdict(dict)

    for model in models_found:
        cp_model_dataset_dir = os.path.join(compound_dir, model, dataset_name)
        if not os.path.isdir(cp_model_dataset_dir):
            continue

        cp_files = [
            f for f in glob.glob(os.path.join(cp_model_dataset_dir, "*.json"))
            if "/not_paper/" not in f and "_raw.json" not in f
        ]

        for cpf in cp_files:
            with open(cpf) as f:
                c_data = json.load(f)

            if not isinstance(c_data, list) or len(c_data) == 0:
                continue

            # Determine num_distractors
            summary = c_data[-1].get("summary", {}) if isinstance(c_data[-1], dict) else {}
            num_distractors = summary.get("num_distractors", None)

            entries = [r for r in c_data if isinstance(r, dict) and r.get("id") is not None]
            if not entries:
                continue

            if num_distractors is None:
                prompt_problems = re.findall(r"Problem \d+:", entries[0].get("original", ""))
                num_distractors = max(0, len(prompt_problems) - 1)

            total_problems = num_distractors + 1
            position = total_problems

            # Compute per-problem ratios across all samples
            ratio_sums = defaultdict(float)
            ratio_counts = defaultdict(int)

            for entry in entries:
                output = entry.get("output", "")
                if not output:
                    continue
                total_len = len(output)
                if total_len == 0:
                    continue

                sections = _segment_output_by_problems(output)
                for p, section_len in sections.items():
                    if p < 1 or p > total_problems:
                        continue
                    ratio_sums[p] += section_len / total_len
                    ratio_counts[p] += 1

            avg_ratios = {}
            for p in range(1, total_problems + 1):
                if ratio_counts.get(p, 0) > 0:
                    avg_ratios[p] = ratio_sums[p] / ratio_counts[p]
                else:
                    avg_ratios[p] = 0.0

            # Normalize so they sum to 1.0
            total_ratio = sum(avg_ratios.values())
            if total_ratio > 0:
                avg_ratios = {p: v / total_ratio for p, v in avg_ratios.items()}

            if position in model_data[model]:
                raise RuntimeError(f"Duplicate position {position} for {model}")
            model_data[model][position] = avg_ratios

    # Add baseline position (position=1, single problem at 100%)
    for model in list(model_data.keys()):
        if 1 not in model_data[model]:
            model_data[model][1] = {1: 1.0}

    # Sort models by name for consistency
    target_order = [
        "openai_gpt-oss-120b",
        "Qwen_Qwen3-30B-A3B-Thinking-2507",
        "nvidia_OpenReasoning-Nemotron-32B",
        "nvidia_OpenReasoning-Nemotron-7B",
    ]
    models_to_plot = [m for m in target_order if m in model_data]
    # Add any remaining models not in target_order
    for m in sorted(model_data.keys()):
        if m not in models_to_plot:
            models_to_plot.append(m)

    if not models_to_plot:
        raise RuntimeError("No models with data to plot")

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    DISTRACTOR_COLOR = '#C0C0C0'  # light grey
    TARGET_COLOR = '#4CAF50'      # green
    BAR_WIDTH = 0.7

    x_ticks = []
    x_tick_labels = []
    model_centers = []
    model_labels = []
    current_x = 0

    for idx, model in enumerate(models_to_plot):
        positions = sorted(model_data[model].keys())
        label_name = shorten(model, MODEL_SHORT_NAMES).replace('\n', ' ')

        current_x_base = current_x
        x_vals = []

        for pos in positions:
            x_val = current_x_base + pos
            x_vals.append(x_val)
            x_ticks.append(x_val)
            x_tick_labels.append(str(pos))

            ratios = model_data[model][pos]
            total_problems = pos

            bottom = 0.0
            for p in range(1, total_problems + 1):
                ratio = ratios.get(p, 0.0)
                pct = ratio * 100.0
                is_target = (p == total_problems)
                color = TARGET_COLOR if is_target else DISTRACTOR_COLOR

                bar = ax.bar(x_val, pct, BAR_WIDTH, bottom=bottom, color=color,
                             edgecolor='black', linewidth=0.5)

                # Annotate if there's enough space (> 4%)
                if pct > 4:
                    ax.text(x_val, bottom + pct / 2, f"{pct:.0f}%",
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            color='black')

                bottom += pct

        if x_vals:
            model_centers.append(sum(x_vals) / len(x_vals))
            model_labels.append(label_name)

        # Gap between model groups
        current_x = x_vals[-1] + 1

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Place model labels underneath
    for center, label in zip(model_centers, model_labels):
        ax.text(center, -0.10, label, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=20, rotation=45)

    ax.set_ylabel("Token Effort (%)", fontsize=22)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DISTRACTOR_COLOR, edgecolor='black', label='Distractor problems'),
        Patch(facecolor=TARGET_COLOR, edgecolor='black', label='Target problem'),
    ]
    ax.legend(handles=legend_elements, ncol=2, fontsize=16, loc='upper right', framealpha=0.8)

    xmax = x_ticks[-1] + 1
    ax.set_xlim(left=0, right=xmax)

    fig.subplots_adjust(bottom=0.35, right=0.95, top=0.95)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"attention_effort_ratios_{dataset_name}.pdf")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Consolidated Optimized Visualization Script")
    parser.add_argument("--experiments_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4_aime_2024")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--plot_type", type=str, required=True, 
                    choices=['accuracy', 'average_accuracy_drop', 
                                 'output_length', 'radar_categories', 'compound', 'attention_effort_ratios'])
    parser.add_argument("--exclude_refusals", action="store_true", help="Exclude refusals from sample pool (use accuracy on attempted)")
    args = parser.parse_args()

    if not args.experiments_dir:
        curr = os.path.dirname(os.path.abspath(__file__))
        experiments_dir = os.path.dirname(curr)
    else:
        experiments_dir = args.experiments_dir

    outdir = args.outdir if args.outdir else os.path.join(experiments_dir, "analysis", "plots")
    safe_dataset = args.dataset.replace('/', '_')

    if args.plot_type == 'compound':
        print(f"Generating compound plot for {args.dataset}...")
        plot_compound(safe_dataset, outdir, experiments_dir)
        return

    if args.plot_type == 'attention_effort_ratios':
        print(f"Generating attention effort ratios plot for {args.dataset}...")
        plot_attention_effort(safe_dataset, outdir, experiments_dir)
        return

    print(f"Loading data for {args.dataset}...")
    metrics_data = load_metrics_data(experiments_dir, safe_dataset)
    rec_data, cond_data = load_recovery_and_conditional(experiments_dir, safe_dataset)

    if args.plot_type == 'accuracy':
        plot_by_model(safe_dataset, metrics_data, outdir, metric='accuracy', failures_on_top=True)
    elif args.plot_type == 'output_length':
        plot_by_model(safe_dataset, metrics_data, outdir, metric='length')
    elif args.plot_type == 'average_accuracy_drop':
        plot_single_metric(safe_dataset, metrics_data, outdir, exclude_refusals=args.exclude_refusals)
    elif args.plot_type == 'radar_categories':
        plot_radar_charts(safe_dataset, metrics_data, outdir)

if __name__ == "__main__":
    main()

