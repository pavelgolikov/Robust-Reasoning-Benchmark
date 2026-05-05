#!/usr/bin/env python3
import argparse
import json
import os
import glob
import re
import matplotlib.lines as mlines
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
    "split_reversal":               "Symbol-Rev",
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

def shorten(name, mapping):
    return mapping.get(name, name)

# ── Fast Data Loading ───────────────────────────────────────────────

def get_latest_json(target_dir):
    json_files = glob.glob(os.path.join(target_dir, "*.json"))
    valid_files = [f for f in json_files if not f.endswith("_raw.json") and "_summary_" not in f and "prompt_recovery" not in f]
    if not valid_files: return None
    return max(valid_files, key=os.path.getmtime)

def load_metrics_data(experiments_dir, safe_dataset):
    data = defaultdict(lambda: defaultdict(dict))
    
    for technique in TECHNIQUE_ORDER:
        tech_dir = os.path.join(experiments_dir, technique, "results")
        if not os.path.isdir(tech_dir): continue
            
        for model in os.listdir(tech_dir):
            if model not in MODEL_SHORT_NAMES: continue
                
            model_dataset_dir = os.path.join(tech_dir, model, safe_dataset)
            if not os.path.isdir(model_dataset_dir): continue
                
            perturb_dir = os.path.join(model_dataset_dir, "perturb")
            target_dir = perturb_dir if os.path.isdir(perturb_dir) else model_dataset_dir
                
            latest_file = get_latest_json(target_dir)
            if not latest_file: continue
                
            try:
                with open(latest_file, 'r') as f:
                    content = json.load(f)
                    
                acc, fail_rate, avg_length = 0.0, 0.0, 0.0
                summary = content[-1].get("summary", {}) if isinstance(content, list) and content else content.get("summary", {}) if isinstance(content, dict) else {}
                results = content if isinstance(content, list) else content.get("results", [])
                    
                avg_length = summary.get("avg_output_tokens", 0.0)
                if summary:
                    acc = summary.get("accuracy", 0.0)
                    if isinstance(acc, float) and 0 < acc <= 1.0: acc *= 100.0
                    
                    refusals = summary.get("refusals", 0)
                    total = summary.get("total", 0)
                    attempted = total - refusals
                    
                    if attempted > 0:
                        acc_attempted = summary.get("accuracy_attempted", summary.get("correct", 0) / attempted)
                        if isinstance(acc_attempted, (float, int)) and 0 < acc_attempted <= 1.0: acc_attempted *= 100.0
                        elif acc_attempted == 0: acc_attempted = 0.0
                    else:
                        acc_attempted = None
                        
                    n_failures = refusals + summary.get("failures", 0)
                    fail_rate = (n_failures / total * 100.0) if total > 0 else 0.0
                else:
                    total_computed, correct, n_refusals_computed, n_failures_computed = 0, 0, 0, 0
                    for r in results:
                        if isinstance(r, dict) and r.get("id") is not None:
                            total_computed += 1
                            if r.get("correct", False): correct += 1
                            if r.get("refusal") is True: n_refusals_computed += 1
                            elif not r.get("correct") and r.get("extracted") is None: n_failures_computed += 1
                                
                    acc = (correct / total_computed * 100.0) if total_computed > 0 else 0.0
                    attempted_computed = total_computed - n_refusals_computed
                    acc_attempted = (correct / attempted_computed * 100.0) if attempted_computed > 0 else None
                    fail_rate = ((n_refusals_computed + n_failures_computed) / total_computed * 100.0) if total_computed > 0 else 0.0
                    total, attempted = total_computed, attempted_computed
                    
                data[technique][model] = {
                    'accuracy': acc, 'accuracy_attempted': acc_attempted,
                    'failure_rate': fail_rate, 'length': avg_length,
                    'total': total, 'attempted': attempted
                }
            except Exception:
                continue
    return data

# ── Plotting ─────────────────────────────────────────────────────────

def get_hatch_style(dataset_name):
    # Use hatch pattern on AIME 2024 as requested
    if "2024" in dataset_name:
        return '///'
    return ''

def plot_by_model(dataset_names, metrics_data_list, outdir, metric='accuracy'):
    if not metrics_data_list: return
    data1 = metrics_data_list[0]
    data2 = metrics_data_list[1] if len(metrics_data_list) > 1 else data1
    dname1 = dataset_names[0]
    dname2 = dataset_names[1] if len(dataset_names) > 1 else dataset_names[0]

    all_models = set()
    for td in data1.values(): all_models.update(td.keys())
    for td in data2.values(): all_models.update(td.keys())
    if not all_models: return

    def _avg_acc(model):
        a1 = [data1[t][model]['accuracy'] for t in data1 if model in data1[t]]
        a2 = [data2[t][model]['accuracy'] for t in data2 if model in data2[t]]
        accs = a1 + a2
        return sum(accs) / len(accs) if accs else 0
    
    all_models = sorted(all_models, key=_avg_acc, reverse=True)
    ordered_techniques = TECHNIQUE_ORDER.copy()
    available = set(data1.keys()).union(set(data2.keys()))
    for t in sorted(available):
        if t not in ordered_techniques: ordered_techniques.append(t)

    n_techniques = len(ordered_techniques)
    tech_labels = [TECHNIQUE_LABELS.get(t, t) for t in ordered_techniques]
    tech_colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(ordered_techniques)}

    hatch1 = get_hatch_style(dname1)
    hatch2 = get_hatch_style(dname2)
    # Ensure they're different if none had "2024"
    if hatch1 == hatch2 and dname1 != dname2: hatch1 = '///'; hatch2 = ''

    def _plot_model_on_ax(ax, model_name):
        accs1, accs2, fails1, fails2, colors, is_mis1, is_mis2 = [], [], [], [], [], [], []
        for t in ordered_techniques:
            td1 = data1.get(t, {})
            td2 = data2.get(t, {})
            
            accs1.append(td1[model_name][metric] if model_name in td1 else 0)
            fails1.append(td1[model_name].get('failure_rate', 0) if model_name in td1 else 0)
            is_mis1.append(model_name not in td1)
            
            accs2.append(td2[model_name][metric] if model_name in td2 else 0)
            fails2.append(td2[model_name].get('failure_rate', 0) if model_name in td2 else 0)
            is_mis2.append(model_name not in td2)
            
            colors.append(tech_colors[t])

        x = np.arange(n_techniques)
        width = 0.4
        
        bars1 = ax.bar(x - width/2, accs1, width, color=colors, edgecolor='black', linewidth=0.5, hatch=hatch1)
        bars2 = ax.bar(x + width/2, accs2, width, color=colors, edgecolor='black', linewidth=0.5, hatch=hatch2)
        
        if metric == 'accuracy':
            ax.bar(x - width/2, fails1, width, bottom=accs1, color=colors, edgecolor='black', linewidth=0.5, alpha=0.4)
            ax.bar(x + width/2, fails2, width, bottom=accs2, color=colors, edgecolor='black', linewidth=0.5, alpha=0.4)

        for b1, b2, a1, a2, m1, m2 in zip(bars1, bars2, accs1, accs2, is_mis1, is_mis2):
            for b, a, m in [(b1, a1, m1), (b2, a2, m2)]:
                if m: text = "N/A"
                else: text = f"{a:.1f}" if 0 < a < 0.5 else f"{a:.0f}"
                if text:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, text, ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)

        ax.set_title(shorten(model_name, MODEL_SHORT_NAMES).replace('\n', ' '), fontsize=15, fontweight='bold', pad=10)
        ax.set_xticks(x); ax.set_xticklabels(tech_labels, fontsize=12, rotation=45, ha='right')
        ax.set_ylabel("Accuracy (%)", fontsize=15)
        ax.set_ylim(0, 115); ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        
        import matplotlib.patches as mpatches
        p1 = mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch1, label=shorten(dname1, DATASET_SHORT_NAMES))
        p2 = mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch2, label=shorten(dname2, DATASET_SHORT_NAMES))
        ax.legend(handles=[p1, p2], loc='upper right', bbox_to_anchor=(1.0, 1.15), fontsize=10)

    ncols, n_models = 2, len(all_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, model_name in enumerate(all_models):
        row, col = divmod(idx, ncols); _plot_model_on_ax(axes[row, col], model_name)
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols); axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.0)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{metric}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_single_metric(dataset_names, metrics_data_list, outdir, exclude_refusals=False):
    if not metrics_data_list: return
    data1 = metrics_data_list[0]
    data2 = metrics_data_list[1] if len(metrics_data_list) > 1 else data1
    dname1 = dataset_names[0]
    dname2 = dataset_names[1] if len(dataset_names) > 1 else dataset_names[0]
    
    all_models = set()
    for td in data1.values(): all_models.update(td.keys())
    for td in data2.values(): all_models.update(td.keys())
    if not all_models: return

    metric_key = 'accuracy_attempted' if exclude_refusals else 'accuracy'
    
    def get_avg_drop(data, model):
        baseline_acc = data.get('baseline', {}).get(model, {}).get(metric_key)
        if baseline_acc is None: return None
        deltas = []
        for t in data:
            if model not in data[t]: continue
            if t == 'baseline': continue
            acc = data[t][model].get(metric_key)
            if acc is not None: deltas.append(baseline_acc - acc)
        return sum(deltas) / len(deltas) if deltas else None

    model_deltas_1 = {m: get_avg_drop(data1, m) for m in all_models}
    model_deltas_2 = {m: get_avg_drop(data2, m) for m in all_models}
    
    # Filter models that have data in both (or at least one)
    plot_models = [m for m in all_models if model_deltas_1.get(m) is not None or model_deltas_2.get(m) is not None]
    plot_models.sort(key=lambda m: ((model_deltas_1.get(m) or 0) + (model_deltas_2.get(m) or 0)) / 2)

    values1 = [model_deltas_1.get(m, 0) for m in plot_models]
    values2 = [model_deltas_2.get(m, 0) for m in plot_models]
    
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(plot_models))]

    hatch1 = get_hatch_style(dname1)
    hatch2 = get_hatch_style(dname2)
    if hatch1 == hatch2 and dname1 != dname2: hatch1 = '///'; hatch2 = ''

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(plot_models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values1, width, color=colors, edgecolor='black', linewidth=0.5, hatch=hatch1)
    bars2 = ax.bar(x + width/2, values2, width, color=colors, edgecolor='black', linewidth=0.5, hatch=hatch2)
    
    for b1, b2, v1, v2 in zip(bars1, bars2, values1, values2):
        for b, v in [(b1, v1), (b2, v2)]:
            # if v < 1 and v > 0:
            if v != 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{np.round(v):.0f}", 
                        ha='center', va='bottom', fontsize=18, fontweight='bold', rotation=0)
            # else:
            #     ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.0f}", 
            #             ha='center', va='bottom', fontsize=14, fontweight='bold', rotation=90)


    labels = [shorten(m, MODEL_SHORT_NAMES).replace('\n', ' ') for m in plot_models]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=18, rotation=45, ha='right')
    ax.set_ylabel("Avg Accuracy Drop (%)", fontsize=20)
    
    import matplotlib.patches as mpatches
    p1 = mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch1, label=shorten(dname1, DATASET_SHORT_NAMES))
    p2 = mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch2, label=shorten(dname2, DATASET_SHORT_NAMES))
    ax.legend(handles=[p1, p2], loc='upper left', fontsize=14)

    all_vals = values1 + values2
    ax.set_ylim(min(all_vals + [0]) * 1.1, max(all_vals + [0]) * 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    suffix = "_no_refusals" if exclude_refusals else ""
    out_path = os.path.join(outdir, f"average_accuracy_drop{suffix}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_radar_charts(dataset_names, metrics_data_list, outdir):
    if not metrics_data_list: return
    data1 = metrics_data_list[0]
    data2 = metrics_data_list[1] if len(metrics_data_list) > 1 else data1
    
    # Average the techniques across the two datasets
    technique_data = defaultdict(lambda: defaultdict(dict))
    
    all_techs = set(data1.keys()).union(set(data2.keys()))
    for t in all_techs:
        models = set(data1.get(t, {}).keys()).union(set(data2.get(t, {}).keys()))
        for m in models:
            a1 = data1.get(t, {}).get(m, {}).get('accuracy', None)
            a2 = data2.get(t, {}).get(m, {}).get('accuracy', None)
            valid = [a for a in [a1, a2] if a is not None]
            if valid:
                technique_data[t][m]['accuracy'] = sum(valid) / len(valid)

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
        
        for angle, label in zip(angles[:-1], CATEGORY_NAMES):
            ha, va = 'center', 'center'
            r_label = 95
            if label == "Syntactic": va = 'top'
            elif label == "Visual": va = 'bottom'
            elif label == "Semantic": ha, va = 'right', 'bottom'; angle -= 0.10
            elif label == "Contextual": ha, va = 'left', 'top'; angle -= 0.10
            ax.text(angle, r_label, label, ha=ha, va=va, fontsize=16, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        ax.set_title(shorten(model, MODEL_SHORT_NAMES).replace('\n', ' '), size=18, fontweight='bold', pad=20)
        ax.set_xticklabels([])
        ax.grid(True, alpha=0.3)

    # Note that this is averaged in the super title
    d_str = " & ".join([shorten(d, DATASET_SHORT_NAMES) for d in dataset_names])
    fig.suptitle(f"Average Performance by Category across {d_str}", fontsize=22, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(outdir, f"radar_categories.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


def load_compound_data_for_dataset(dataset_name, experiments_dir):
    compound_dir = os.path.join(experiments_dir, "compound", "results")
    baseline_dir = os.path.join(experiments_dir, "baseline", "results")
    if not os.path.isdir(compound_dir): return {}
    models_found = [d for d in os.listdir(compound_dir) if os.path.isdir(os.path.join(compound_dir, d))]
    
    model_data = defaultdict(dict)
    for model in models_found:
        bl_model_dataset_dir = os.path.join(baseline_dir, model, dataset_name, "compound")
        if os.path.isdir(bl_model_dataset_dir):
            bl_files = glob.glob(os.path.join(bl_model_dataset_dir, "*.json"))
            if bl_files:
                b_correct, b_total, b_cutoffs = 0, 0, 0
                for blf in bl_files:
                    with open(blf) as f: b_data = json.load(f)
                    summary = b_data[-1].get("summary", {}) if isinstance(b_data, list) and b_data else b_data.get("summary", {}) if isinstance(b_data, dict) else {}
                    correct = summary.get("correct", 0)
                    total = summary.get("total", 0)
                    cutoffs = summary.get("max_token_cutoffs", 0)
                    if total == 0 and isinstance(b_data, list): total = len([item for item in b_data if isinstance(item, dict) and "id" in item])
                    b_correct += correct; b_total += total; b_cutoffs += cutoffs
                
                b_acc = (b_correct / float(b_total)) if b_total > 0 else 0
                if b_total == 0:
                    with open(bl_files[0]) as f:
                        fallback = json.load(f)
                        b_acc = fallback[-1].get("summary", {}).get("accuracy", 0.0) if isinstance(fallback, list) else fallback.get("summary", {}).get("accuracy", 0.0)
                        b_cutoffs = fallback[-1].get("summary", {}).get("max_token_cutoffs", 0) if isinstance(fallback, list) else fallback.get("summary", {}).get("max_token_cutoffs", 0)
                
                if isinstance(b_acc, float) and 0 < b_acc <= 1.0: b_acc *= 100.0
                model_data[model][1] = {'acc': float(b_acc), 'cutoffs': b_cutoffs}
        
        cp_model_dataset_dir = os.path.join(compound_dir, model, dataset_name)
        if os.path.isdir(cp_model_dataset_dir):
            cp_files = glob.glob(os.path.join(cp_model_dataset_dir, "*.json"))
            pos_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'cutoffs': 0})
            for cpf in cp_files:
                with open(cpf) as f: c_data = json.load(f)
                last_item = c_data[-1] if isinstance(c_data, list) and c_data else c_data
                first_item = c_data[0] if isinstance(c_data, list) and len(c_data) > 0 else c_data
                summary = last_item.get("summary", {})
                correct = summary.get("correct", 0)
                total = summary.get("total", 0)
                cutoffs = summary.get("max_token_cutoffs", 0)
                if total == 0 and isinstance(c_data, list): total = len([item for item in c_data if isinstance(item, dict) and "id" in item])
                orig = first_item.get("original", "")
                distractors = max(0, len(re.findall(r'Problem \d+:', orig)) - 1)
                if distractors == 0 and model in ["claude-opus-4-6", "gpt-5.4", "gemini-3.1-pro-preview"]: distractors = 3
                if distractors < 1: continue
                pos = distractors + 1
                pos_stats[pos]['correct'] += correct
                pos_stats[pos]['total'] += total
                pos_stats[pos]['cutoffs'] += cutoffs
                
            for pos, stats in pos_stats.items():
                c_acc = (stats['correct'] / float(stats['total'])) if stats['total'] > 0 else summary.get("accuracy", 0.0)
                if isinstance(c_acc, float) and 0 < c_acc <= 1.0: c_acc *= 100.0
                model_data[model][pos] = {'acc': float(c_acc), 'cutoffs': stats['cutoffs']}
    return model_data

def plot_compound(dataset_names, outdir, experiments_dir):
    data1 = load_compound_data_for_dataset(dataset_names[0], experiments_dir)
    data2 = load_compound_data_for_dataset(dataset_names[1], experiments_dir) if len(dataset_names) > 1 else {}
    
    # AIME 2024 gets circle, AIME 2025 (or other) gets square as requested
    marker1 = 'o' if '2024' in dataset_names[0] else 's'
    marker2 = 'o' if len(dataset_names) > 1 and '2024' in dataset_names[1] else 's'
    if marker1 == marker2 and len(dataset_names) > 1:
        marker2 = 's' if marker1 == 'o' else 'o'
    
    target_models = [
        "gemini-3.1-pro-preview", "gpt-5.4", "claude-opus-4-6",
        "Qwen_Qwen3-30B-A3B-Thinking-2507", "nvidia_OpenReasoning-Nemotron-32B",
        "nvidia_OpenReasoning-Nemotron-7B", "openai_gpt-oss-120b",
        "deepseek-ai_DeepSeek-R1-Distill-Llama-70B"
    ]
    
    all_models = set(data1.keys()).union(set(data2.keys()))
    models_to_plot = [m for m in target_models if m in all_models and (data1.get(m) or data2.get(m))]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x_ticks, x_tick_labels, model_centers, model_labels = [], [], [], []
    current_x = 0
    
    for idx, model in enumerate(models_to_plot):
        color = PALETTE[idx % len(PALETTE)]
        label_name = shorten(model, MODEL_SHORT_NAMES).replace('\n', ' ')
        
        pos1 = sorted(data1.get(model, {}).keys())
        pos2 = sorted(data2.get(model, {}).keys())
        
        # We need a unified X axis block for this model
        all_pos = sorted(list(set(pos1).union(set(pos2))))
        if not all_pos: continue
        
        current_x_base = current_x
        x_vals = []
        for p in all_pos:
            x_val = current_x_base + p
            x_vals.append(x_val)
            x_ticks.append(x_val)
            x_tick_labels.append(str(p))
            
        model_centers.append(sum(x_vals) / len(x_vals))
        model_labels.append(label_name)
        
        # Plot line 1
        if pos1:
            x1 = [current_x_base + p for p in pos1]
            y1 = [data1[model][p]['acc'] for p in pos1]
            ax.plot(x1, y1, marker=marker1, markersize=8, linewidth=3, linestyle='-', color=color)
            
            # Annotate first and last point cutoffs
            first_p, last_p = pos1[0], pos1[-1]
            first_val, last_val = data1[model][first_p].get('cutoffs', 0), data1[model][last_p].get('cutoffs', 0)
            ax.text(x1[0], y1[0] - 2, f"{first_val}", fontsize=14, ha='center', va='top', color=color, fontweight='bold')
            ax.text(x1[-1], y1[-1] + 2, f"{last_val}", fontsize=14, ha='center', va='bottom', color=color, fontweight='bold')
        
        # Plot line 2
        if pos2:
            x2 = [current_x_base + p for p in pos2]
            y2 = [data2[model][p]['acc'] for p in pos2]
            ax.plot(x2, y2, marker=marker2, markersize=8, linewidth=3, linestyle='-', color=color)
            
            # Annotate first and last point cutoffs
            first_p, last_p = pos2[0], pos2[-1]
            first_val, last_val = data2[model][first_p].get('cutoffs', 0), data2[model][last_p].get('cutoffs', 0)
            ax.text(x2[0], y2[0] - 2, f"{first_val}", fontsize=14, ha='center', va='top', color=color, fontweight='bold')
            ax.text(x2[-1], y2[-1] + 2, f"{last_val}", fontsize=14, ha='center', va='bottom', color=color, fontweight='bold')
            
        current_x = x_vals[-1] + 2  
                
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    for center, label in zip(model_centers, model_labels):
        ax.text(center, -0.12, label, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=22, rotation=45)
                
    ax.set_ylabel("Accuracy on Last Problem (%)", fontsize=22)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Legend - small and focused on datasets only
    legend_elements = [
        mlines.Line2D([], [], color='gray', marker=marker1, linestyle='-', markersize=10, label=shorten(dataset_names[0], DATASET_SHORT_NAMES)),
    ]
    if len(dataset_names) > 1:
        legend_elements.append(mlines.Line2D([], [], color='gray', marker=marker2, linestyle='-', markersize=10, label=shorten(dataset_names[1], DATASET_SHORT_NAMES)))
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=16, title="Datasets", title_fontsize=18)
    
    fig.subplots_adjust(bottom=0.35, right=0.75, top=0.92)
    xmax = x_ticks[-1] + 1 if x_ticks else 10
    ax.set_xlim(left=0, right=xmax)
    
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"compound.pdf")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Combined Dataset Visualization")
    parser.add_argument("--experiments_dir", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="HuggingFaceH4_aime_2024,MathArena_aime_2025")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--plot_type", type=str, required=True, 
                    choices=['accuracy', 'average_accuracy_drop', 'output_length', 'radar_categories', 'compound'])
    parser.add_argument("--exclude_refusals", action="store_true")
    args = parser.parse_args()

    experiments_dir = args.experiments_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = args.outdir or os.path.join(experiments_dir, "analysis", "plots")
    
    datasets = [d.strip() for d in args.datasets.split(',')]
    safe_datasets = [d.replace('/', '_') for d in datasets]

    if args.plot_type == 'compound':
        print(f"Generating combined compound plot...")
        plot_compound(safe_datasets, outdir, experiments_dir)
        return

    print(f"Loading data for datasets: {safe_datasets}...")
    metrics_data_list = [load_metrics_data(experiments_dir, sd) for sd in safe_datasets]

    if args.plot_type == 'accuracy':
        plot_by_model(safe_datasets, metrics_data_list, outdir, metric='accuracy')
    elif args.plot_type == 'output_length':
        print("Output length combined plot is paused. Skipping.")
    elif args.plot_type == 'average_accuracy_drop':
        plot_single_metric(safe_datasets, metrics_data_list, outdir, exclude_refusals=args.exclude_refusals)
    elif args.plot_type == 'radar_categories':
        plot_radar_charts(safe_datasets, metrics_data_list, outdir)

if __name__ == "__main__":
    main()
