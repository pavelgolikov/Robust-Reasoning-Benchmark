#!/usr/bin/env python3
"""
Compute rebuttal-facing statistics from saved experiment JSON files.

This script intentionally uses only the Python standard library. It does not
load HuggingFace datasets, vLLM, pandas, or model tokenizers.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
from collections import defaultdict


TRANSFORMATION_NAMES = [
    "not_not",
    "opposites",
    "wrappers",
    "interleaved_context_line",
    "interleaved_context_word",
    "interleaved_context_symbol",
    "sentence_reversal",
    "word_reversal",
    "split_reversal",
    "rail_fence",
    "rectangle_perimeter",
    "snake_vertical",
    "snake_horizontal",
]

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def result_entries(data):
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        return [], {}

    entries = []
    summary = {}
    for item in data:
        if isinstance(item, dict) and "summary" in item and len(item) == 1:
            summary = item["summary"] or {}
        elif isinstance(item, dict):
            entries.append(item)
    return entries, summary


def clean_percent(value):
    if value is None:
        return ""
    return f"{100.0 * value:.2f}"


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return None, None
    p = successes / total
    denom = 1.0 + (z * z / total)
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def problem_subset_interval(problem_scores, n_draws=10000, seed=42):
    """Resample problem IDs and recompute the mean each time."""
    values = list(problem_scores.values())
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low_idx = int(0.025 * (n_draws - 1))
    high_idx = int(0.975 * (n_draws - 1))
    return means[low_idx], means[high_idx]


def parse_path(path, experiments_dir):
    rel = os.path.relpath(path, experiments_dir)
    parts = rel.split(os.sep)
    if len(parts) < 5 or parts[1] != "results":
        return None

    technique = parts[0]
    model = parts[2]
    dataset = parts[3]
    file_name = os.path.basename(path)
    subcondition = ""
    if technique == "baseline" and len(parts) >= 6:
        subcondition = parts[4]
    return technique, model, dataset, subcondition, file_name


def infer_decode_name(file_name):
    for name in TRANSFORMATION_NAMES:
        if f"_{name}_decode_recovery_" in file_name:
            return name
    return "unknown"


def infer_condition(technique, subcondition, file_name, summary):
    task = summary.get("task", "")
    if technique == "baseline":
        return f"baseline:{subcondition or 'unknown'}"
    if task == "decode_only_recovery" or technique == "decode_recovery":
        return f"decode_recovery:{infer_decode_name(file_name)}"
    if task == "equal_length_passive_context" or technique == "passive_context":
        num_distractors = summary.get("num_distractors", "")
        suffix = f":d{num_distractors}" if num_distractors != "" else ""
        return f"passive_context{suffix}"
    return technique


def score_key_for(entries, summary, technique):
    task = summary.get("task", "")
    if task == "decode_only_recovery" or technique == "decode_recovery":
        return "recovered"
    if any("recovered" in entry for entry in entries):
        return "recovered"
    return "correct"


def entry_success(entry, score_key):
    return bool(entry.get(score_key, False))


def entry_failure(entry, score_key):
    if score_key == "recovered":
        if entry.get("has_recovered_tags") is False:
            return True
        return not bool(entry.get("recovered", False))
    extracted = entry.get("extracted")
    return extracted is None or (isinstance(extracted, str) and extracted.startswith("ERROR"))


def cutoff_limit(summary):
    if summary.get("max_tokens") is not None:
        return summary["max_tokens"]
    return summary.get("max_model_length")


def entry_cutoff(entry, summary):
    limit = cutoff_limit(summary)
    if limit is None or entry.get("output_tokens") is None:
        return False
    try:
        return float(entry["output_tokens"]) >= float(limit) * 0.98
    except (TypeError, ValueError):
        return False


def summarize_file(path, experiments_dir, n_draws, seed):
    parsed = parse_path(path, experiments_dir)
    if not parsed:
        return None
    technique, model, dataset, subcondition, file_name = parsed

    data = read_json(path)
    entries, summary = result_entries(data)
    if not entries:
        return None

    condition = infer_condition(technique, subcondition, file_name, summary)
    score_key = score_key_for(entries, summary, technique)

    successes = sum(entry_success(entry, score_key) for entry in entries)
    total = len(entries)
    failures = sum(entry_failure(entry, score_key) for entry in entries)
    cutoffs = sum(entry_cutoff(entry, summary) for entry in entries)
    if cutoffs == 0 and isinstance(summary.get("max_token_cutoffs"), int):
        cutoffs = summary["max_token_cutoffs"]

    by_problem = defaultdict(list)
    for entry in entries:
        if entry.get("id") is None:
            continue
        by_problem[str(entry["id"])].append(1.0 if entry_success(entry, score_key) else 0.0)

    problem_scores = {
        problem_id: sum(values) / len(values)
        for problem_id, values in by_problem.items()
        if values
    }
    problem_mean = (
        sum(problem_scores.values()) / len(problem_scores)
        if problem_scores
        else successes / total
    )
    problem_low, problem_high = problem_subset_interval(problem_scores, n_draws=n_draws, seed=seed)
    sample_low, sample_high = wilson_interval(successes, total)

    sample_counts = [len(values) for values in by_problem.values()]

    return {
        "path": path,
        "file": file_name,
        "technique": technique,
        "condition": condition,
        "model": model,
        "dataset": dataset,
        "score_key": score_key,
        "successes": successes,
        "total": total,
        "rate": successes / total if total else None,
        "sample_interval_low": sample_low,
        "sample_interval_high": sample_high,
        "problem_count": len(problem_scores),
        "problem_mean": problem_mean,
        "problem_interval_low": problem_low,
        "problem_interval_high": problem_high,
        "min_samples_per_problem": min(sample_counts) if sample_counts else 0,
        "max_samples_per_problem": max(sample_counts) if sample_counts else 0,
        "mean_samples_per_problem": (
            sum(sample_counts) / len(sample_counts)
            if sample_counts
            else 0.0
        ),
        "failures": failures,
        "cutoffs": cutoffs,
        "max_tokens": summary.get("max_tokens", ""),
        "max_model_length": summary.get("max_model_length", ""),
        "temperature": summary.get("temperature", ""),
        "top_p": summary.get("top_p", ""),
        "n_samples": summary.get("n_samples", ""),
        "num_distractors": summary.get("num_distractors", ""),
        "summary_task": summary.get("task", ""),
        "mtime": os.path.getmtime(path),
        "problem_scores": problem_scores,
    }


def discover_result_files(experiments_dir):
    pattern = os.path.join(experiments_dir, "*", "results", "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    keep = []
    for path in files:
        name = os.path.basename(path)
        if name.endswith("_raw.json"):
            continue
        if name.startswith(("jobs_", "tracking_", "batch_")):
            continue
        if "_summary_" in name:
            continue
        keep.append(path)
    return keep


def keep_latest_by_key(summaries):
    latest = {}
    for row in summaries:
        key = (row["model"], row["dataset"], row["condition"])
        if key not in latest or row["mtime"] > latest[key]["mtime"]:
            latest[key] = row
    return list(latest.values())


def filter_rows(rows, models, datasets, conditions):
    def allowed(value, allowlist):
        return not allowlist or value in allowlist

    out = []
    for row in rows:
        if not allowed(row["model"], models):
            continue
        if not allowed(row["dataset"], datasets):
            continue
        if not allowed(row["condition"], conditions) and not allowed(row["technique"], conditions):
            continue
        out.append(row)
    return out


def paired_comparison(ref, other, n_draws, seed, label):
    common = sorted(set(ref["problem_scores"]) & set(other["problem_scores"]))
    diffs = {
        problem_id: other["problem_scores"][problem_id] - ref["problem_scores"][problem_id]
        for problem_id in common
    }
    if common:
        mean_diff = sum(diffs.values()) / len(diffs)
        low, high = problem_subset_interval(diffs, n_draws=n_draws, seed=seed)
        worse = sum(1 for value in diffs.values() if value < 0)
        better = sum(1 for value in diffs.values() if value > 0)
        same = sum(1 for value in diffs.values() if value == 0)
    else:
        mean_diff, low, high = None, None, None
        worse, better, same = 0, 0, 0

    return {
        "comparison": label,
        "model": other["model"],
        "dataset": other["dataset"],
        "reference_condition": ref["condition"],
        "condition": other["condition"],
        "common_problem_count": len(common),
        "reference_problem_mean": ref["problem_mean"],
        "condition_problem_mean": other["problem_mean"],
        "problem_mean_difference": mean_diff,
        "difference_interval_low": low,
        "difference_interval_high": high,
        "problems_better": better,
        "problems_same": same,
        "problems_worse": worse,
        "reference_file": ref["file"],
        "condition_file": other["file"],
    }


def build_comparisons(rows, n_draws, seed):
    by_key = {(row["model"], row["dataset"], row["condition"]): row for row in rows}
    comparisons = []

    for row in rows:
        model = row["model"]
        dataset = row["dataset"]
        condition = row["condition"]
        technique = row["technique"]

        if technique in TRANSFORMATION_NAMES:
            ref = by_key.get((model, dataset, "baseline:perturb"))
            if ref:
                comparisons.append(
                    paired_comparison(ref, row, n_draws, seed, "perturbation_vs_baseline")
                )
        elif condition == "compound":
            ref = by_key.get((model, dataset, "baseline:compound"))
            if ref:
                comparisons.append(
                    paired_comparison(ref, row, n_draws, seed, "compound_vs_baseline")
                )
        elif condition.startswith("passive_context"):
            ref = by_key.get((model, dataset, "baseline:compound"))
            if ref:
                comparisons.append(
                    paired_comparison(ref, row, n_draws, seed, "passive_context_vs_baseline")
                )
            ref = by_key.get((model, dataset, "compound"))
            if ref:
                comparisons.append(
                    paired_comparison(ref, row, n_draws, seed, "passive_context_vs_compound")
                )

    return comparisons


SUMMARY_COLUMNS = [
    "model",
    "dataset",
    "condition",
    "score_key",
    "successes",
    "total",
    "rate",
    "sample_interval_low",
    "sample_interval_high",
    "problem_count",
    "problem_mean",
    "problem_interval_low",
    "problem_interval_high",
    "min_samples_per_problem",
    "max_samples_per_problem",
    "mean_samples_per_problem",
    "failures",
    "cutoffs",
    "max_tokens",
    "max_model_length",
    "temperature",
    "top_p",
    "n_samples",
    "num_distractors",
    "file",
    "path",
]

COMPARISON_COLUMNS = [
    "comparison",
    "model",
    "dataset",
    "reference_condition",
    "condition",
    "common_problem_count",
    "reference_problem_mean",
    "condition_problem_mean",
    "problem_mean_difference",
    "difference_interval_low",
    "difference_interval_high",
    "problems_better",
    "problems_same",
    "problems_worse",
    "reference_file",
    "condition_file",
]


def write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_json(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean_rows = []
    for row in rows:
        clean_rows.append({key: value for key, value in row.items() if key != "problem_scores"})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_rows, f, indent=2)


def print_compact_table(rows, limit):
    if not rows:
        print("No result rows found.")
        return

    header = (
        f"{'model':<42} {'dataset':<26} {'condition':<34} "
        f"{'count':>10} {'rate%':>8} {'problem range%':>20} {'cutoffs':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows[:limit]:
        problem_range = ""
        if row["problem_interval_low"] is not None:
            problem_range = (
                f"{clean_percent(row['problem_interval_low'])}-"
                f"{clean_percent(row['problem_interval_high'])}"
            )
        print(
            f"{row['model']:<42} {row['dataset']:<26} {row['condition']:<34} "
            f"{row['successes']:>4}/{row['total']:<5} {clean_percent(row['rate']):>8} "
            f"{problem_range:>20} {row['cutoffs']:>8}"
        )


def parse_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Compute raw counts, cutoff counts, and problem-level uncertainty ranges for rebuttal."
    )
    parser.add_argument(
        "--experiments_dir",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Path to the experiments directory.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to experiments/analysis/rebuttal_stats.",
    )
    parser.add_argument("--models", default="", help="Comma-separated safe model names to keep.")
    parser.add_argument("--datasets", default="", help="Comma-separated safe dataset names to keep.")
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated condition names or technique directories to keep.",
    )
    parser.add_argument(
        "--all_files",
        action="store_true",
        help="Keep all result files instead of only the newest file per model/dataset/condition.",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=10000,
        help="Number of repeated problem-subset calculations for interval estimates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_limit", type=int, default=40)
    args = parser.parse_args()

    experiments_dir = os.path.abspath(args.experiments_dir)
    out_dir = args.out_dir or os.path.join(experiments_dir, "analysis", "rebuttal_stats")

    summaries = []
    for path in discover_result_files(experiments_dir):
        try:
            row = summarize_file(path, experiments_dir, args.draws, args.seed)
        except Exception as exc:
            print(f"Warning: failed to summarize {path}: {exc}")
            continue
        if row:
            summaries.append(row)

    if not args.all_files:
        summaries = keep_latest_by_key(summaries)

    summaries = filter_rows(
        summaries,
        models=set(parse_list(args.models)),
        datasets=set(parse_list(args.datasets)),
        conditions=set(parse_list(args.conditions)),
    )
    summaries.sort(key=lambda row: (row["model"], row["dataset"], row["condition"]))

    comparisons = build_comparisons(summaries, args.draws, args.seed)
    comparisons.sort(key=lambda row: (row["model"], row["dataset"], row["comparison"], row["condition"]))

    summary_csv = os.path.join(out_dir, "summary_counts.csv")
    comparison_csv = os.path.join(out_dir, "paired_comparisons.csv")
    summary_json = os.path.join(out_dir, "summary_counts.json")
    comparison_json = os.path.join(out_dir, "paired_comparisons.json")

    write_csv(summary_csv, summaries, SUMMARY_COLUMNS)
    write_csv(comparison_csv, comparisons, COMPARISON_COLUMNS)
    write_json(summary_json, summaries)
    write_json(comparison_json, comparisons)

    print(f"Wrote {len(summaries)} summary rows to {summary_csv}")
    print(f"Wrote {len(comparisons)} paired comparison rows to {comparison_csv}")
    print_compact_table(summaries, args.print_limit)


if __name__ == "__main__":
    main()
