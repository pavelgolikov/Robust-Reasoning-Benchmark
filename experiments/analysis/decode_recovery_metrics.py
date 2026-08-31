#!/usr/bin/env python3
"""
Re-score saved decode-only recovery runs.

The run script reports a single binary gate (character error rate <= 0.02), which
collapses near-perfect reconstructions and copied transformed input into the same
bucket. This script re-scores the stored `recovered_text` post-hoc, so every run --
old or new -- is measured identically and nothing has to be generated again. It
reports the metric set plan.md asks for: normalized exact match, character error
rate (as a distribution, not just a pass/fail rate), ordered n-gram overlap, and
the transformation-specific residual statuses.

Standard library only: no HuggingFace datasets, vLLM, pandas, or model tokenizers.
"""

import argparse
import csv
import difflib
import functools
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_decode_recovery import (  # noqa: E402
    TRANSFORMATION_NAMES,
    char_error_rate,
    normalize_for_recovery,
)


NGRAM_ORDERS = (1, 2, 4)

# Presentation-only LaTeX differences. Models routinely reproduce a problem exactly but
# re-render $x$ as \(x \), $$...$$ as \[...\], or add \left/\right -- worth 10-40% character
# error rate while changing nothing mathematically. Scoring through these is what separates
# "failed to decode" from "decoded, then reformatted".
_MATH_DELIMITERS = (r"\(", r"\)", r"\[", r"\]", "$$", "$")
_PRESENTATION_MACROS = (
    r"\left", r"\right", r"\bullet", r"\quad", r"\qquad", r"\!", r"\,", r"\;", r"\:",
)


def normalize_for_scoring(text):
    """normalize_for_recovery, plus insensitivity to LaTeX math presentation.

    Math delimiters, spacing macros and the ';' that newline-flattening inserts during
    dataset sanitization are dropped from both sides. Case, digits, variable names,
    operators and LaTeX structure are all preserved, so a genuine decoding error still
    registers.
    """
    text = normalize_for_recovery(text)
    for macro in _PRESENTATION_MACROS:
        text = re.sub(re.escape(macro) + r"(?![a-zA-Z])", " ", text)
    for token in _MATH_DELIMITERS:
        text = text.replace(token, " ")
    text = text.replace(";", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_ignoring_space(text):
    """normalize_for_scoring with all whitespace removed.

    Where a space sits inside a formula ("8!= 40320" vs "8! = 40320") is presentation like
    the delimiters above, and it accounts for a further ~11 points on Nemotron-32B. Reported
    as its own column rather than folded into the headline number, since collapsing word
    boundaries could in principle hide a genuine error.
    """
    return re.sub(r"\s+", "", normalize_for_scoring(text))


# Exact Levenshtein is O(len(a)*len(b)) in pure Python. The run script only switches to a
# difflib ratio above 6000 characters, which is fine when scoring is amortized over
# generation but far too slow for re-scoring ~37k stored samples. Keep the exact distance
# for the normal case (an AIME problem is ~300 characters) and fall back earlier for the
# rambling outputs, where the exact value would not change any conclusion.
CER_EXACT_AREA_LIMIT = 1_000_000


@functools.lru_cache(maxsize=200_000)
def bounded_char_error_rate(reference, hypothesis):
    ref_len = max(1, len(reference))
    # A hypothesis several times longer than the reference is already saturated: the edit
    # distance is at least the length difference, so the rate exceeds 2 whatever the exact
    # alignment is. Return that lower bound instead of aligning a 100k-character ramble.
    if len(hypothesis) > 3 * ref_len:
        return (len(hypothesis) - len(reference)) / ref_len, True
    if ref_len * len(hypothesis) > CER_EXACT_AREA_LIMIT:
        ratio = difflib.SequenceMatcher(None, reference, hypothesis).ratio()
        return 1.0 - ratio, True
    return char_error_rate(reference, hypothesis)


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
        elif isinstance(item, dict) and "recovered_text" in item:
            entries.append(item)
    return entries, summary


def ngram_counts(tokens, order):
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i:i + order]) for i in range(len(tokens) - order + 1))


def ngram_overlap(reference_tokens, hypothesis_tokens, order):
    """Ordered n-gram precision/recall/F1 of hypothesis against reference."""
    ref = ngram_counts(reference_tokens, order)
    hyp = ngram_counts(hypothesis_tokens, order)
    if not ref and not hyp:
        return 1.0, 1.0, 1.0
    if not ref or not hyp:
        return 0.0, 0.0, 0.0

    matched = sum((ref & hyp).values())
    precision = matched / sum(hyp.values())
    recall = matched / sum(ref.values())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def mean(values):
    return sum(values) / len(values) if values else None


def infer_transform(file_name):
    for name in TRANSFORMATION_NAMES:
        if f"_{name}_decode_recovery_" in file_name:
            return name
    return "unknown"


def parse_path(path, experiments_dir):
    rel = os.path.relpath(path, experiments_dir)
    parts = rel.split(os.sep)
    if len(parts) < 5 or parts[0] != "decode_recovery" or parts[1] != "results":
        return None
    return parts[2], parts[3], os.path.basename(path)


def score_entry(entry, threshold):
    """Re-score one sample from its stored recovered text.

    Two scorings are reported side by side: `raw` reproduces what the run script recorded,
    `norm` additionally ignores LaTeX presentation. The gap between them is the share of
    "failures" that were really reformatting.
    """
    original = entry.get("canonical_original", "")
    recovered = entry.get("recovered_text", "")

    # The as-run column must reproduce the run script exactly, so take its own stored values
    # rather than recomputing them through a different distance approximation.
    if "char_error_rate" in entry:
        raw_cer = float(entry["char_error_rate"])
        raw_estimated = bool(entry.get("char_error_rate_estimated", False))
        raw_exact = bool(entry.get("exact_normalized_match", False))
        raw_recovered_flag = bool(entry.get("recovered", raw_exact or raw_cer <= threshold))
    else:
        raw_original = normalize_for_recovery(original)
        raw_recovered_text = normalize_for_recovery(recovered)
        raw_cer, raw_estimated = bounded_char_error_rate(raw_original, raw_recovered_text)
        raw_exact = raw_recovered_text == raw_original
        raw_recovered_flag = bool(raw_exact or raw_cer <= threshold)

    norm_original = normalize_for_scoring(original)
    norm_recovered = normalize_for_scoring(recovered)
    norm_cer, _ = bounded_char_error_rate(norm_original, norm_recovered)
    norm_exact = norm_recovered == norm_original

    nospace_original = normalize_ignoring_space(original)
    nospace_recovered = normalize_ignoring_space(recovered)
    nospace_cer, _ = bounded_char_error_rate(nospace_original, nospace_recovered)

    ref_tokens = norm_original.split()
    hyp_tokens = norm_recovered.split()
    overlaps = {}
    for order in NGRAM_ORDERS:
        precision, recall, f1 = ngram_overlap(ref_tokens, hyp_tokens, order)
        overlaps[order] = {"precision": precision, "recall": recall, "f1": f1}

    return {
        "raw_exact": raw_exact,
        "raw_cer": raw_cer,
        "raw_recovered": raw_recovered_flag,
        "cer_estimated": raw_estimated,
        "norm_exact": norm_exact,
        "norm_cer": norm_cer,
        "norm_recovered": bool(norm_exact or norm_cer <= threshold),
        "nospace_recovered": bool(nospace_original == nospace_recovered or nospace_cer <= threshold),
        "overlaps": overlaps,
        "residual_status": entry.get("residual_status", "unknown"),
        "has_tags": bool(entry.get("has_recovered_tags", False)),
        "output_tokens": entry.get("output_tokens"),
        "id": entry.get("id"),
    }


def summarize_file(path, experiments_dir, threshold):
    parsed = parse_path(path, experiments_dir)
    if not parsed:
        return None
    model, dataset, file_name = parsed

    entries, summary = result_entries(read_json(path))
    if not entries:
        return None

    scored = [score_entry(entry, threshold) for entry in entries]
    residual_counts = Counter(s["residual_status"] for s in scored)

    by_problem = defaultdict(list)
    for s in scored:
        if s["id"] is not None:
            by_problem[str(s["id"])].append(1.0 if s["norm_recovered"] else 0.0)

    max_tokens = summary.get("max_tokens") or summary.get("max_model_length")
    cutoffs = 0
    if max_tokens:
        cutoffs = sum(
            1 for s in scored
            if s["output_tokens"] is not None
            and float(s["output_tokens"]) >= 0.98 * float(max_tokens)
        )

    raw_cers = [s["raw_cer"] for s in scored]
    norm_cers = [s["norm_cer"] for s in scored]

    row = {
        "model": model,
        "dataset": dataset,
        "transformation": infer_transform(file_name),
        "total": len(scored),
        "recovered_rate_as_run": mean([1.0 if s["raw_recovered"] else 0.0 for s in scored]),
        "exact_match_rate_as_run": mean([1.0 if s["raw_exact"] else 0.0 for s in scored]),
        "cer_mean_as_run": mean(raw_cers),
        "recovered_rate": mean([1.0 if s["norm_recovered"] else 0.0 for s in scored]),
        "exact_match_rate": mean([1.0 if s["norm_exact"] else 0.0 for s in scored]),
        "cer_mean": mean(norm_cers),
        "cer_p25": quantile(norm_cers, 0.25),
        "cer_p50": quantile(norm_cers, 0.50),
        "cer_p75": quantile(norm_cers, 0.75),
        # A sample with no <RECOVERED_PROBLEM> tags falls back to the whole output, reasoning
        # trace included, so it scores as a decode failure when it is really a failure to follow
        # the output protocol. Report compliance and the rate among compliant samples separately.
        "tagged_rate": mean([1.0 if s["has_tags"] else 0.0 for s in scored]),
        "recovered_rate_tagged": mean(
            [1.0 if s["norm_recovered"] else 0.0 for s in scored if s["has_tags"]]
        ),
        "recovered_rate_ignoring_space": mean([1.0 if s["nospace_recovered"] else 0.0 for s in scored]),
        "missing_tags": sum(1 for s in scored if not s["has_tags"]),
        "cutoffs": cutoffs,
        "problem_count": len(by_problem),
        "cer_threshold": threshold,
        "file": file_name,
        "path": path,
    }

    for order in NGRAM_ORDERS:
        row[f"ngram{order}_precision"] = mean([s["overlaps"][order]["precision"] for s in scored])
        row[f"ngram{order}_recall"] = mean([s["overlaps"][order]["recall"] for s in scored])
        row[f"ngram{order}_f1"] = mean([s["overlaps"][order]["f1"] for s in scored])

    for status in sorted(residual_counts):
        row[f"residual_{status}"] = residual_counts[status]

    return row


COLUMNS = [
    "model",
    "dataset",
    "transformation",
    "total",
    "recovered_rate",
    "recovered_rate_tagged",
    "recovered_rate_ignoring_space",
    "tagged_rate",
    "exact_match_rate",
    "cer_mean",
    "cer_p25",
    "cer_p50",
    "cer_p75",
    "ngram1_precision",
    "ngram1_recall",
    "ngram1_f1",
    "ngram2_precision",
    "ngram2_recall",
    "ngram2_f1",
    "ngram4_precision",
    "ngram4_recall",
    "ngram4_f1",
    "recovered_rate_as_run",
    "exact_match_rate_as_run",
    "cer_mean_as_run",
    "missing_tags",
    "cutoffs",
    "problem_count",
    "cer_threshold",
    "residual_recovered",
    "residual_partial_or_other",
    "residual_missing_recovered_tags",
    "residual_closer_to_transformed_than_original",
    "residual_copied_transformed_input",
    "residual_contains_transformation_markers",
    "residual_empty_recovered_text",
    "file",
    "path",
]


def write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def print_table(rows, limit):
    if not rows:
        print("No decode-recovery result rows found.")
        return
    header = (
        f"{'model':<34} {'ds':<22} {'transform':<27} "
        f"{'rec%':>6} {'+tag':>6} {'+sp':>6} {'as-run':>7} {'CERp50':>7} {'1gF1':>6} {'4gF1':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows[:limit]:
        print(
            f"{row['model'][:34]:<34} {row['dataset'][:22]:<22} {row['transformation']:<27} "
            f"{100 * row['recovered_rate']:>6.1f} {100 * (row['recovered_rate_tagged'] or 0):>6.1f} "
            f"{100 * row['recovered_rate_ignoring_space']:>6.1f} {100 * row['recovered_rate_as_run']:>7.1f} "
            f"{row['cer_p50']:>7.3f} {row['ngram1_f1']:>6.3f} {row['ngram4_f1']:>6.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Re-score decode-only recovery runs post hoc.")
    parser.add_argument(
        "--experiments_dir",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--models", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument(
        "--cer_threshold",
        type=float,
        default=0.02,
        help="Character error rate at or below which a sample counts as recovered (run default: 0.02).",
    )
    parser.add_argument("--print_limit", type=int, default=100)
    args = parser.parse_args()

    experiments_dir = os.path.abspath(args.experiments_dir)
    out_dir = args.out_dir or os.path.join(experiments_dir, "analysis", "rebuttal_stats")

    keep_models = {m.strip() for m in args.models.split(",") if m.strip()}
    keep_datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}

    pattern = os.path.join(experiments_dir, "decode_recovery", "results", "**", "*.json")
    rows = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        if os.path.basename(path).endswith("_raw.json"):
            continue
        try:
            row = summarize_file(path, experiments_dir, args.cer_threshold)
        except Exception as exc:
            print(f"Warning: failed to score {path}: {exc}")
            continue
        if not row:
            continue
        if keep_models and row["model"] not in keep_models:
            continue
        if keep_datasets and row["dataset"] not in keep_datasets:
            continue
        rows.append(row)

    rows.sort(key=lambda r: (r["model"], r["dataset"], r["transformation"]))

    out_csv = os.path.join(out_dir, "decode_recovery_metrics.csv")
    write_csv(out_csv, rows, COLUMNS)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print_table(rows, args.print_limit)


if __name__ == "__main__":
    main()
