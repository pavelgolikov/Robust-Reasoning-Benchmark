"""
Analyze compound experiment outputs to detect how models structure their
responses across multiple problems.  For each result JSON in the compound/
directory, we detect "Problem X" markers in the model output and compare
against the number of distractors that were used.
"""

import os
import re
import json
import glob


def find_result_files(compound_dir):
    """Return all non-raw result JSON files under compound/results/."""
    results_dir = os.path.join(compound_dir, "results")
    pattern = os.path.join(results_dir, "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    # Exclude raw checkpoints, tracking files, jobs files
    files = [
        f for f in files
        if not os.path.basename(f).startswith("batch_tracking")
        and not os.path.basename(f).startswith("jobs_")
        and "_raw.json" not in f
        and "/not_paper/" not in f
    ]
    # Ignore non-target models
    ignore_models = ["ministral", "limo", "falcon", "gpt-5.4", "gemini", "claude", "deepseek"]
    files = [
        f for f in files
        if not any(m in f.lower() for m in ignore_models)
    ]
    return sorted(files)


def extract_model_name(filepath):
    """Extract model name from the path: compound/results/<model>/..."""
    parts = filepath.split(os.sep)
    try:
        idx = parts.index("results")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return os.path.basename(filepath)


def segment_output_by_problems(output, total_problems):
    """
    Split the output into sections based on "Problem X" markers.
    Returns a dict: {problem_number: section_text}.
    Only considers the FIRST occurrence of each "Problem X" as a section boundary.
    Text before the first marker is assigned to a 'preamble'.
    """
    # Find all "Problem <number>" positions (first occurrence of each)
    pattern = re.compile(r"Problem\s*(\d+)")
    seen = {}
    for m in pattern.finditer(output):
        p = int(m.group(1))
        if p not in seen:
            seen[p] = m.start()

    if not seen:
        return {}

    # Sort by position
    sorted_markers = sorted(seen.items(), key=lambda x: x[1])

    sections = {}
    for i, (p, start) in enumerate(sorted_markers):
        if i + 1 < len(sorted_markers):
            end = sorted_markers[i + 1][1]
        else:
            end = len(output)
        sections[p] = output[start:end]

    return sections


def analyze_file(filepath):
    """
    Parse a single result JSON.  Returns a dict with:
      - model: str
      - num_distractors: int  (from summary, or inferred from prompts)
      - total_problems_in_prompt: int  (distractors + 1 target)
      - entries: list of per-sample dicts with problem marker counts and token ratios
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Last element is the summary dict
    summary = None
    entries = []
    for item in data:
        if "summary" in item:
            summary = item["summary"]
        elif "output" in item:
            entries.append(item)

    num_distractors = summary.get("num_distractors", None) if summary else None

    # If num_distractors not in summary, infer from first entry's prompt
    if num_distractors is None and entries:
        prompt_problems = re.findall(r"Problem \d+:", entries[0]["original"])
        num_distractors = max(0, len(prompt_problems) - 1)

    total_problems = (num_distractors or 0) + 1

    per_sample = []
    for entry in entries:
        output = entry.get("output", "")
        # Find all "Problem X" occurrences (with or without colon)
        markers = re.findall(r"Problem\s*(\d+)", output)
        marker_counts = {}
        for m in markers:
            marker_counts[int(m)] = marker_counts.get(int(m), 0) + 1

        # Segment output and compute per-problem token ratios
        sections = segment_output_by_problems(output, total_problems)
        total_len = len(output) if output else 1  # avoid division by zero
        token_ratios = {}
        for p, section_text in sections.items():
            token_ratios[p] = len(section_text) / total_len

        per_sample.append({
            "id": entry.get("id"),
            "correct": entry.get("correct"),
            "marker_counts": marker_counts,
            "total_markers": len(markers),
            "unique_problems_mentioned": len(marker_counts),
            "token_ratios": token_ratios,
        })

    return {
        "model": extract_model_name(filepath),
        "file": filepath,
        "num_distractors": num_distractors,
        "total_problems_in_prompt": total_problems,
        "num_entries": len(per_sample),
        "per_sample": per_sample,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    compound_dir = os.path.join(base_dir, "..", "compound")

    files = find_result_files(compound_dir)
    if not files:
        raise FileNotFoundError(f"No result JSON files found under {compound_dir}/results/")

    print(f"Found {len(files)} result file(s) in compound/results/\n")

    for filepath in files:
        info = analyze_file(filepath)
        model = info["model"]
        n_dist = info["num_distractors"]
        total_p = info["total_problems_in_prompt"]
        n_entries = info["num_entries"]

        # Aggregate across all samples
        all_unique = [s["unique_problems_mentioned"] for s in info["per_sample"]]
        all_total = [s["total_markers"] for s in info["per_sample"]]

        # Count how many samples mention each problem number
        # and accumulate token ratios
        problem_mention_counts = {}
        problem_token_ratio_sums = {}
        problem_token_ratio_counts = {}
        for s in info["per_sample"]:
            for p in s["marker_counts"]:
                problem_mention_counts[p] = problem_mention_counts.get(p, 0) + 1
            for p, ratio in s["token_ratios"].items():
                problem_token_ratio_sums[p] = problem_token_ratio_sums.get(p, 0.0) + ratio
                problem_token_ratio_counts[p] = problem_token_ratio_counts.get(p, 0) + 1

        avg_unique = sum(all_unique) / len(all_unique) if all_unique else 0
        avg_total = sum(all_total) / len(all_total) if all_total else 0

        print(f"{'='*70}")
        print(f"Model: {model}")
        print(f"File:  {os.path.basename(filepath)}")
        print(f"Distractors: {n_dist} | Total problems in prompt: {total_p} | Samples: {n_entries}")
        print(f"Avg unique 'Problem X' mentioned per output: {avg_unique:.1f}")
        print(f"Avg total 'Problem X' markers per output:    {avg_total:.1f}")
        print(f"Problem mention frequency and token effort:")
        all_problems = sorted(set(list(problem_mention_counts.keys()) + list(problem_token_ratio_sums.keys())))
        for p in all_problems:
            mention_count = problem_mention_counts.get(p, 0)
            pct = 100 * mention_count / n_entries if n_entries else 0
            avg_ratio = 100 * problem_token_ratio_sums.get(p, 0) / problem_token_ratio_counts[p] if problem_token_ratio_counts.get(p, 0) > 0 else 0
            is_target = " <-- TARGET" if p == total_p else ""
            print(f"  Problem {p}: {mention_count}/{n_entries} samples ({pct:.0f}%) - {avg_ratio:.0f}% token effort{is_target}")
        print()


if __name__ == "__main__":
    main()

