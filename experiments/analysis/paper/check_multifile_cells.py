"""Audit behind the fix to rebuttal_statistics.py (pooling multi-file cells).

Several proprietary-model cells were run in two batches, so their results are split across two
JSON files. rebuttal_statistics.py used to keep only the newest file per (model, dataset, condition)
and silently discarded the rest. This script shows, for every proprietary cell:

  * how many result files it has and how many samples they hold together,
  * how many samples summary_counts.csv now counts for it (should equal the total),
  * for cells with two files, whether the files are independent samples or a duplicated re-run:
    the number of (problem id, full output text) pairs that appear in both files. A re-run of the same
    samples would share almost all of them. Fixed placeholder outputs written by the API wrapper
    ("ERROR: Refused by model safety filter", "ERROR: Could not parse ...") and empty outputs are
    identical by construction, so they are counted separately from genuine model text.

Run:  python3 experiments/analysis/paper/check_multifile_cells.py
"""

import csv
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(os.path.dirname(HERE))
SUMMARY = os.path.join(EXPERIMENTS, "analysis", "rebuttal_stats", "summary_counts.csv")
PROPRIETARY = ("claude-opus-4-6", "gpt-5.4", "gemini-3.1-pro-preview")
SKIP_PREFIX = ("jobs_", "tracking_", "batch_")


def records(path):
    data = json.load(open(path))
    recs = data if isinstance(data, list) else data.get("results", [])
    return [r for r in recs if isinstance(r, dict) and "id" in r]


def main():
    cells = defaultdict(list)          # (transformation dir, model, dataset, arm) -> [paths]
    for path in glob.glob(os.path.join(EXPERIMENTS, "*", "results", "*", "*", "**", "*.json"), recursive=True):
        parts = os.path.relpath(path, EXPERIMENTS).split(os.sep)
        name = os.path.basename(path)
        if parts[2] not in PROPRIETARY or name.startswith(SKIP_PREFIX) or "_summary_" in name:
            continue
        arm = parts[4] if len(parts) > 5 else ""
        cells[(parts[0], parts[2], parts[3], arm)].append(path)

    # summary_counts.csv records the first file of a pooled cell in 'path'; each cell is looked up by
    # whichever of its files appears there.
    counted = {}
    with open(SUMMARY, newline="") as f:
        for r in csv.DictReader(f):
            counted[os.path.basename(r.get("path") or "")] = int(r["total"])

    print("%-28s %-24s %-5s %-9s %5s %8s %8s %13s %13s" %
          ("transformation", "model", "data", "arm", "files", "on disk", "counted",
           "shared model", "shared error"))
    multi = short = genuine_shared = 0
    for key in sorted(cells):
        paths = sorted(cells[key])
        recs = [records(p) for p in paths]
        total = sum(len(r) for r in recs)
        used = next((counted[os.path.basename(p)] for p in paths if os.path.basename(p) in counted), None)
        shared = placeholder = "-"
        if len(paths) == 2:
            multi += 1
            a = {(str(r["id"]), r.get("output") or "") for r in recs[0]}
            b = {(str(r["id"]), r.get("output") or "") for r in recs[1]}
            both = a & b
            fixed = {x for x in both if not x[1].strip() or x[1].startswith("ERROR:")}
            shared, placeholder = str(len(both - fixed)), str(len(fixed))
            genuine_shared += len(both - fixed)
        if total != 240:
            short += 1
        print("%-28s %-24s %-5s %-9s %5d %8d %8s %13s %13s" %
              (key[0][:28], key[1][:24], key[2][-4:], key[3] or "-", len(paths), total,
               used if used is not None else "?", shared, placeholder))
    print("\ncells: %d   split across two files: %d   not exactly 8 x 30 = 240 on disk: %d"
          % (len(cells), multi, short))
    print("genuine model outputs duplicated across the two files of a cell: %d" % genuine_shared)


if __name__ == "__main__":
    main()
