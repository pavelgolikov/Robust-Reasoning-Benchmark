
import json
import argparse
import os
import re
import glob
import time
from collections import defaultdict
from typing import List, Dict, Any

import torch
from tqdm import tqdm

# ── Configuration (mirrored from plot_results.py) ────────────────────

TECHNIQUE_LABELS = {
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
    "rectangle_perimeter":          "Rectangle Perimeter",
}

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
}

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#64B5CD", "#CCB974",
]

# Ordered list of techniques for analysis (no baseline / context_saturation)
TECHNIQUES_LIST = [
    'not_not',
    'opposites',
    'sentence_reversal',
    'split_reversal',
    'word_reversal',
    'wrappers',
    'interleaved_context_line',
    'interleaved_context_word',
    'interleaved_context_symbol',
    'rail_fence',
    'rectangle_perimeter',
]


# ── Text helpers ─────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Basic normalization: remove latex, extra whitespace."""
    if not text:
        return ""
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
    text = text.replace('$', '').replace('\\', '')
    return " ".join(text.split())

def make_windows(tokens: List[str], window_size: int, step_size: int = 10) -> List[str]:
    """Create sliding windows of text from tokens."""
    windows = []
    if not tokens:
        return []
    if len(tokens) <= window_size:
        return [" ".join(tokens)]

    for i in range(0, len(tokens) - window_size + 1, step_size):
        window = tokens[i : i + window_size]
        windows.append(" ".join(window))

    if len(tokens) > window_size:
        last_window = tokens[-window_size:]
        windows.append(" ".join(last_window))

    return list(set(windows))


# ── File discovery ───────────────────────────────────────────────────

def find_latest_result(experiment_name: str, model_name: str, dataset_name: str, base_dir: str) -> str:
    """Finds the latest JSON result file for a given experiment/model/dataset."""
    safe_model = model_name.replace('/', '_').replace(' ', '_')
    safe_dataset = dataset_name.replace('/', '_')
    results_dir = os.path.join(base_dir, experiment_name, "results", safe_model, safe_dataset)

    if not os.path.exists(results_dir):
        return None

    files = glob.glob(os.path.join(results_dir, "*.json"))
    files = [f for f in files if not f.endswith("_prompt_recovery.json")
             and "semantic" not in f and not f.endswith("_raw.json")]

    if not files:
        return None
    return max(files, key=os.path.getmtime)


def discover_models(base_dir: str, techniques: List[str], dataset_name: str) -> List[str]:
    """Auto-discover all model names that have results in any of the given techniques."""
    safe_dataset = dataset_name.replace('/', '_')
    models = set()
    for technique in techniques:
        results_dir = os.path.join(base_dir, technique, "results")
        if not os.path.isdir(results_dir):
            continue
        for model_dir_name in os.listdir(results_dir):
            dataset_dir = os.path.join(results_dir, model_dir_name, safe_dataset)
            if os.path.isdir(dataset_dir):
                json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
                json_files = [f for f in json_files if not f.endswith("_prompt_recovery.json")]
                if json_files:
                    models.add(model_dir_name)
    return sorted(models)


# ── Semantic analysis ────────────────────────────────────────────────

def analyze_single_file(result_file: str, model, args) -> Dict[str, Any]:
    if args.dry:
        return {
            "source_file": result_file,
            "total_samples": 100,
            "original_correct": 10,
            "semantic_correct": 20,
            "recovered_cases": [{"id": 0, "score": 0.99, "target": "DRY RUN", "best_window": "DRY RUN"}],
            "original_accuracy": 0.1,
            "semantic_accuracy": 0.2,
            "note": "DRY RUN - MOCK DATA"
        }

    with open(result_file, 'r') as f:
        data = json.load(f)

    # Handle dict-with-results format
    if isinstance(data, dict) and "results" in data:
        data = data["results"]

    total = len(data)
    summary = {
        "source_file": result_file,
        "total_samples": total,
        "original_correct": 0,
        "semantic_correct": 0,
        "recovered_cases": []
    }

    # ── Phase 1: Collect all targets and windows (CPU) ──
    entries_to_analyze = []  # list of (entry_idx, entry, norm_target, windows_list)

    for entry in tqdm(data, desc="  Collecting windows", leave=False):
        is_orig_correct = entry.get('correct', False)
        if is_orig_correct:
            summary["original_correct"] += 1
            summary["semantic_correct"] += 1
            continue

        target_text = entry.get('unmodified_original', '')
        model_output = entry.get('output', '')

        norm_target = normalize_text(target_text)
        target_tokens = norm_target.split()
        target_len = len(target_tokens)

        norm_output = normalize_text(model_output)
        output_tokens = norm_output.split()

        if not norm_target or not norm_output:
            continue

        window_sizes = [int(target_len * 0.8), target_len, int(target_len * 1.2)]

        all_windows = []
        for w_size in window_sizes:
            if w_size < 1: w_size = 1
            all_windows.extend(make_windows(output_tokens, window_size=w_size, step_size=args.step_size))

        if not all_windows:
            continue

        entries_to_analyze.append((entry, norm_target, all_windows))

    if not entries_to_analyze:
        summary["original_accuracy"] = summary["original_correct"] / total if total > 0 else 0
        summary["semantic_accuracy"] = summary["semantic_correct"] / total if total > 0 else 0
        return summary

    # ── Phase 2: Batch-encode all targets and windows (GPU) ──
    from sentence_transformers import util

    all_targets = [norm_target for _, norm_target, _ in entries_to_analyze]
    all_flat_windows = []
    window_offsets = []  # (start_idx, end_idx) into all_flat_windows for each entry
    for _, _, windows in entries_to_analyze:
        start = len(all_flat_windows)
        all_flat_windows.extend(windows)
        window_offsets.append((start, len(all_flat_windows)))

    print(f"    Batch encoding: {len(all_targets)} targets, {len(all_flat_windows)} windows...")
    target_embeddings = model.encode(all_targets, convert_to_tensor=True,
                                     show_progress_bar=False, batch_size=256)
    window_embeddings = model.encode(all_flat_windows, convert_to_tensor=True,
                                     show_progress_bar=True, batch_size=256)

    # ── Phase 3: Compute similarities per entry ──
    for i, (entry, norm_target, windows) in enumerate(tqdm(entries_to_analyze, desc="  Computing similarities", leave=False)):
        start, end = window_offsets[i]
        target_emb = target_embeddings[i].unsqueeze(0)
        win_embs = window_embeddings[start:end]

        cosine_scores = util.cos_sim(target_emb, win_embs)[0]

        best_idx = int(cosine_scores.argmax())
        max_score = float(cosine_scores[best_idx])

        if max_score >= args.threshold:
            summary["semantic_correct"] += 1
            summary["recovered_cases"].append({
                "id": entry.get('id'),
                "score": max_score,
                "target": norm_target,
                "best_window": windows[best_idx]
            })

    summary["original_accuracy"] = summary["original_correct"] / total if total > 0 else 0
    summary["semantic_accuracy"] = summary["semantic_correct"] / total if total > 0 else 0

    return summary


# ── Plotting ─────────────────────────────────────────────────────────

def scan_recovery_results(base_dir: str, techniques: List[str], safe_dataset: str):
    """
    Scan prompt_recovery JSON reports that were previously generated.
    Returns: data[technique][model] = {'recovery_rate': pct, 'n_samples': n}
    """
    report_base = os.path.join(base_dir, "analysis", "prompt_reconstruction", "results")

    data = defaultdict(dict)

    if not os.path.isdir(report_base):
        return data

    # Walk: results/{model}/{dataset}/*_prompt_recovery*.json
    for model_dir_name in sorted(os.listdir(report_base)):
        model_dataset_dir = os.path.join(report_base, model_dir_name, safe_dataset)
        if not os.path.isdir(model_dataset_dir):
            continue

        for fname in sorted(os.listdir(model_dataset_dir)):
            if not fname.endswith('.json') or 'prompt_recovery' not in fname:
                continue

            # Match technique from filename: {technique}_prompt_recovery_{timestamp}.json
            matched_technique = None
            for t in techniques:
                if fname.startswith(t + "_prompt_recovery"):
                    matched_technique = t
                    break

            if not matched_technique:
                continue

            fpath = os.path.join(model_dataset_dir, fname)
            try:
                with open(fpath) as f:
                    report = json.load(f)

                sem_acc = report.get('semantic_accuracy', 0) * 100  # to pct
                n = report.get('total_samples', 0)

                # Keep latest report per technique/model (filenames sort by timestamp)
                data[matched_technique][model_dir_name] = {
                    'recovery_rate': sem_acc, 'n_samples': n
                }
            except Exception as e:
                print(f"  Warning: could not read {fpath}: {e}")

    return data


def plot_recovery(technique_data, dataset_name, outdir):
    """
    Produce a bar-chart grid of prompt recovery rates, mirroring plot_results.py style.
    technique_data: dict[technique] -> dict[model] -> {recovery_rate, n_samples}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    all_models = set()
    for td in technique_data.values():
        all_models.update(td.keys())
    all_models = sorted(all_models)

    if not all_models:
        print("  No recovery data found. Run analysis first.")
        return

    techniques_with_data = [t for t in TECHNIQUES_LIST if t in technique_data and technique_data[t]]
    n_techniques = len(techniques_with_data)
    if n_techniques == 0:
        print("  No techniques with recovery data found.")
        return

    model_colors = {}
    for i, model in enumerate(all_models):
        model_colors[model] = PALETTE[i % len(PALETTE)]

    ncols = min(4, n_techniques)
    nrows = (n_techniques + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    if n_techniques == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    dataset_label = DATASET_SHORT_NAMES.get(dataset_name, dataset_name)
    fig.suptitle(f"Prompt Recovery Rate — {dataset_label}", fontsize=20, fontweight='bold', y=0.98)

    for idx, technique in enumerate(techniques_with_data):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        td = technique_data[technique]

        subplot_models = [m for m in all_models if m in td]
        if not subplot_models:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(TECHNIQUE_LABELS.get(technique, technique), fontsize=13, fontweight='bold', pad=8)
            continue

        x = np.arange(len(subplot_models))
        bar_width = 0.65
        rates = [td[m]['recovery_rate'] for m in subplot_models]
        colors = [model_colors[m] for m in subplot_models]

        labels = []
        for m in subplot_models:
            short = MODEL_SHORT_NAMES.get(m, m)
            n = td[m]['n_samples']
            n_str = str(int(n)) if n == int(n) else f"{n:.1f}"
            labels.append(f"{short}\n(n={n_str})")

        bars = ax.bar(x, rates, bar_width, color=colors, edgecolor='white', linewidth=0.5)

        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                    f"{rate:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

        title = TECHNIQUE_LABELS.get(technique, technique)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax.set_ylabel("Recovery Rate (%)", fontsize=11)
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(n_techniques, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"prompt_recovery_{dataset_name}.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved plot: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", type=str, default='all',
                        help="Comma-separated technique names or 'all'")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. GAIR/LIMO-v2) or 'all' to auto-discover")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--dry", action="store_true", help="Dry run: mock analysis")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip analysis, just plot from existing recovery JSONs")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip model/technique combos that already have a recovery JSON")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.dirname(os.path.dirname(script_dir))
    base_dir = experiments_dir

    safe_dataset = args.dataset.replace('/', '_')

    # Determine techniques
    if args.names == 'all':
        techniques = TECHNIQUES_LIST
    else:
        techniques = [n.strip() for n in args.names.split(',') if n.strip()]

    # ── Plot-only mode ──
    if args.plot_only:
        print("Plot-only mode: scanning existing recovery JSONs...")
        recovery_data = scan_recovery_results(base_dir, techniques, safe_dataset)
        if not recovery_data:
            print("No recovery reports found. Run analysis first.")
            return
        outdir = os.path.join(base_dir, "analysis", "plots")
        plot_recovery(recovery_data, safe_dataset, outdir)
        return

    # ── Determine models ──
    if args.model == 'all':
        models = discover_models(base_dir, techniques, args.dataset)
        print(f"Auto-discovered {len(models)} models: {models}")
    else:
        models = [args.model.replace('/', '_').replace(' ', '_')]

    if not models:
        print("No models found.")
        return

    print(f"Techniques: {techniques}")
    print(f"Models: {models}")
    print(f"Base Directory: {base_dir}")

    # ── Load embedding model ──
    if args.dry:
        print("\n--- DRY RUN: MOCKING ANALYSIS ---")
        embed_model = None
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer model on {device}...")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(args.embedding_model, device=device)

    # ── Run analysis per model ──
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        output_dir = os.path.join(base_dir, "analysis", "prompt_reconstruction", "results",
                                  model_name, safe_dataset)
        os.makedirs(output_dir, exist_ok=True)

        table_rows = []

        for technique_name in techniques:
            print(f"\n  Processing: {technique_name}")

            # Check if recovery report already exists
            if args.skip_existing:
                existing = glob.glob(os.path.join(output_dir, f"{technique_name}_prompt_recovery_*.json"))
                if existing:
                    print(f"    Skipping (existing report found)")
                    continue

            latest_file = find_latest_result(technique_name, model_name, args.dataset, base_dir)
            if not latest_file:
                print(f"    No result file found. Skipping.")
                continue

            print(f"    Source: {os.path.basename(latest_file)}")

            summary = analyze_single_file(latest_file, embed_model, args)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if args.dry:
                output_filename = f"{technique_name}_prompt_recovery_DRYRUN.json"
            else:
                output_filename = f"{technique_name}_prompt_recovery_{timestamp}.json"

            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"    Saved: {output_path}")

            table_rows.append({
                "name": technique_name,
                "total": summary["total_samples"],
                "orig_acc": summary["original_accuracy"],
                "sem_acc": summary["semantic_accuracy"],
                "recovered": len(summary["recovered_cases"]),
                "file": os.path.basename(latest_file)
            })

        # Print summary table for this model
        if table_rows:
            header = f"{'Experiment':<30} | {'Total':<8} | {'Orig Acc':<10} | {'Sem Acc':<10} | {'Recovered':<10} | {'File'}"
            divider = "-" * len(header)
            print("\n" + header)
            print(divider)
            for row in table_rows:
                print(f"{row['name']:<30} | {row['total']:<8} | {row['orig_acc']:<10.2%} | {row['sem_acc']:<10.2%} | {row['recovered']:<10} | {row['file']}")

    # ── Generate plot from all collected results ──
    print("\n\nGenerating prompt recovery plot...")
    recovery_data = scan_recovery_results(base_dir, techniques, safe_dataset)
    if recovery_data:
        outdir = os.path.join(base_dir, "analysis", "plots")
        plot_recovery(recovery_data, safe_dataset, outdir)
    else:
        print("No recovery data found to plot.")

    # Append to summary file
    summary_file = os.path.join(base_dir, "analysis", "prompt_reconstruction", "prompt_recovery_analysis.txt")
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_file, "a") as f:
        f.write(f"\n\nAnalysis Run: {timestamp_str} (Models: {models}, Dataset: {args.dataset})\n")
    print(f"\nSummary appended to: {summary_file}")


if __name__ == "__main__":
    main()
