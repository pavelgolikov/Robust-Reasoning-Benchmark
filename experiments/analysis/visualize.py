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
    "context_saturation":           "Context",
    "rectangle_perimeter":          "Rectangle",
    "snake_vertical":               "Snake-V",
    "snake_horizontal":             "Snake-H",
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

CATEGORIES = {
    "Syntactic": ["split_reversal", "word_reversal", "sentence_reversal"],
    "Semantic": ["not_not", "opposites", "wrappers"],
    "Visual": ["rail_fence"],
    "Contextual": ["interleaved_context_line", "interleaved_context_symbol", "interleaved_context_word", "context_saturation"],
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
    return f"{value/1000:.1f}"

# ── Fast Data Loading ───────────────────────────────────────────────

def load_accuracy_and_length(experiments_dir, safe_dataset):
    """Load data from analysis/output_length/results/ summaries."""
    summary_base = os.path.join(experiments_dir, "analysis", "output_length", "results")
    data = defaultdict(lambda: defaultdict(dict))
    
    if not os.path.isdir(summary_base):
        print(f"Error: Summary directory not found at {summary_base}")
        return data

    for model_name in os.listdir(summary_base):
        model_dataset_dir = os.path.join(summary_base, model_name, safe_dataset)
        if not os.path.isdir(model_dataset_dir):
            continue
        
        for f in os.listdir(model_dataset_dir):
            if f.endswith(".json") and "_summary_" in f:
                technique = f.split("_summary_")[0]
                fpath = os.path.join(model_dataset_dir, f)
                try:
                    with open(fpath) as j:
                        summary = json.load(j)
                    
                    canonical_model = model_name
                    if model_name == "HAIR_LIMO-v2": canonical_model = "GAIR_LIMO-v2"
                    
                    data[technique][canonical_model] = {
                        'accuracy': summary.get('accuracy', 0),
                        'failure_rate': summary.get('failure_rate', 0),
                        'n_samples': summary.get('n_samples', 0),
                        'length': summary.get('length', 0)
                    }
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
    out_path = os.path.join(outdir, f"{metric}.pdf")
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
    out_path = os.path.join(outdir, "prompt_recovery.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_single_metric(dataset_name, technique_data, outdir):
    all_models = set()
    for td in technique_data.values(): all_models.update(td.keys())
    if not all_models: return

    def _avg_acc(model):
        accs = [technique_data[t][model]['accuracy'] for t in technique_data if model in technique_data[t]]
        return sum(accs) / len(accs) if accs else 0
    all_models = sorted(all_models, key=_avg_acc, reverse=True)

    model_deltas = {}
    for model in all_models:
        baseline_acc = technique_data.get('baseline', {}).get(model, {}).get('accuracy')
        if baseline_acc is None: continue
        deltas = [baseline_acc - technique_data[t][model]['accuracy'] for t in technique_data if t != 'baseline' and model in technique_data[t]]
        if deltas: model_deltas[model] = sum(deltas) / len(deltas)

    plot_models = sorted(model_deltas.keys(), key=lambda m: model_deltas[m])
    values = [model_deltas[m] for m in plot_models]
    colors = [PALETTE[all_models.index(m) % len(PALETTE)] for m in plot_models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(np.arange(len(plot_models)), values, 0.65, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.5 if val >= 0 else -1.5), f"{val:.0f}", \
            ha='center', va='bottom' if val >= 0 else 'top', fontsize=18, fontweight='bold')

    dataset_label = shorten(dataset_name, DATASET_SHORT_NAMES)
    # ax.set_title(f"Average Accuracy Drop — {dataset_label}", fontsize=22, fontweight='bold', pad=20)
    ax.set_xticks(np.arange(len(plot_models)))
    ax.set_xticklabels([shorten(m, MODEL_SHORT_NAMES).replace('\n', ' ') for m in plot_models], fontsize=18, rotation=45, ha='right')
    ax.set_ylabel("Avg Accuracy Drop (%)", fontsize=20)
    ax.set_ylim(min(values + [0]) * 1.1, max(values + [0]) * 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(outdir, "average_accuracy_drop.pdf")
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
        ax.set_rgrids([20, 40, 60, 80], labels=["20", "40", "60", "80"], fontsize=8, color='gray')
        
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
    out_path = os.path.join(outdir, "radar_categories.pdf")
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
    out_path = os.path.join(outdir, "conditional_accuracy.pdf")
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
    r1 = ax.bar(x - width/2, [d['base'] for d in plot_data], width, label='Baseline', color='#4C72B0', edgecolor='black', linewidth=0.5)
    r2 = ax.bar(x + width/2, [d['g_cond'] for d in plot_data], width, label='Global Cond.', color='#DD8452', edgecolor='black', linewidth=0.5)
    
    # ax.set_title(f"Global Reasoning Stability — {shorten(dataset_name, DATASET_SHORT_NAMES)}", fontsize=22, fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=20)
    ax.set_xticks(x); ax.set_xticklabels([shorten(m, MODEL_SHORT_NAMES).replace('\n', ' ') for m in models], rotation=45, ha='right', fontsize=20)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.8); ax.set_ylim(0, 112); ax.grid(axis='y', alpha=0.3)
    
    for r in list(r1) + list(r2):
        ax.annotate(f'{r.get_height():.0f}', xy=(r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Highlight Open Source Models
    os_keywords = ["LIMO", "Falcon", "DeepSeek", "Qwen", "gpt-oss"]
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
    out_path = os.path.join(outdir, "global_conditional_accuracy.pdf")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Consolidated Optimized Visualization Script")
    parser.add_argument("--experiments_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4_aime_2024")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--plot_type", type=str, required=True, 
                        choices=['accuracy', 'global_conditional_accuracy', 'average_accuracy_drop', 
                                 'output_length', 'prompt_recovery', 'radar_categories', 'conditional_accuracy'])
    args = parser.parse_args()

    if not args.experiments_dir:
        curr = os.path.dirname(os.path.abspath(__file__))
        experiments_dir = os.path.dirname(curr)
    else:
        experiments_dir = args.experiments_dir

    outdir = args.outdir if args.outdir else os.path.join(experiments_dir, "analysis", "plots")
    safe_dataset = args.dataset.replace('/', '_')

    print(f"Loading data for {args.dataset}...")
    acc_len_data = load_accuracy_and_length(experiments_dir, safe_dataset)
    rec_data, cond_data = load_recovery_and_conditional(experiments_dir, safe_dataset)

    if args.plot_type == 'accuracy':
        plot_by_model(safe_dataset, acc_len_data, outdir, metric='accuracy', failures_on_top=True)
    elif args.plot_type == 'output_length':
        plot_by_model(safe_dataset, acc_len_data, outdir, metric='length')
    elif args.plot_type == 'average_accuracy_drop':
        plot_single_metric(safe_dataset, acc_len_data, outdir)
    elif args.plot_type == 'radar_categories':
        plot_radar_charts(safe_dataset, acc_len_data, outdir)
    elif args.plot_type == 'prompt_recovery':
        plot_recovery(rec_data, safe_dataset, outdir, accuracy_data=acc_len_data, accuracy_overlay=True)
    elif args.plot_type == 'conditional_accuracy':
        plot_conditional_accuracy(safe_dataset, cond_data, acc_len_data, outdir)
    elif args.plot_type == 'global_conditional_accuracy':
        plot_global_conditional_accuracy(safe_dataset, cond_data, acc_len_data, outdir)

if __name__ == "__main__":
    main()
