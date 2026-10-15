#!/usr/bin/env python3
"""Paper figure: decode-only recovery against solve accuracy for a single model.

The multi-panel `--plot_type decode_recovery` view in visualize.py covers all three models
and is what the appendix table is built from. This is the main-text version: one model, one
panel per dataset, transformations ordered by granularity rather than by TECHNIQUE_ORDER,
because granularity is the variable the result is about.

Usage:
    python3 experiments/analysis/plot_decode_recovery_paper.py [--model ...] [--outdir ...]
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Coarse -> fine. Mirrors the Unit column of Table 1 in the paper. The first four are the
# block where the model both attempts (stub rate 0.0%) and transcribes verbatim, so the
# binary decode/solve comparison is like-for-like; the rest are in the appendix figure.
COARSE_N = 4
GRANULARITY_ORDER = [
    ("sentence_reversal", "Sentence\nRev."),
    ("interleaved_context_line", "Interleave\n(Line)"),
    ("opposites", "Opposites"),
    ("wrappers", "Wrappers"),
    ("word_reversal", "Word\nRev."),
    ("interleaved_context_word", "Interleave\n(Word)"),
    ("split_reversal", "Symbol\nRev."),
    ("snake_horizontal", "Snake\n(Horiz.)"),
    ("rectangle_perimeter", "Rect.\nPerim."),
    ("snake_vertical", "Snake\n(Vert.)"),
    ("interleaved_context_symbol", "Interleave\n(Sym.)"),
    ("rail_fence", "Rail\nFence"),
]
DATASETS = [("HuggingFaceH4_aime_2024", "AIME 2024"), ("MathArena_aime_2025", "AIME 2025")]

# Matches the PALETTE convention in visualize.py / visualize_combined.py.
C_SOLVE = "#636363"
C_DECODE = "#4C72B0"
STUB_THRESHOLD = 10.0


def read_rows(path):
    if not os.path.isfile(path):
        raise SystemExit(
            f"{path} not found. Generate it first:\n"
            f"  python3 experiments/analysis/decode_recovery_metrics.py\n"
            f"  bash experiments/analysis/run_rebuttal_statistics.sh"
        )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_dir", default=os.path.join(here, "rebuttal_stats"))
    ap.add_argument("--outdir", default=os.path.join(here, "plots"))
    ap.add_argument("--model", default="Qwen_Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--name", default="decode_recovery")
    ap.add_argument("--subset", choices=("coarse", "all"), default="coarse",
                    help="coarse = the four like-for-like transforms (main text); all = full gradient.")
    ap.add_argument("--show_fidelity", action="store_true",
                    help="Also plot graded 4-gram fidelity, which explains the gate dipping below solve.")
    args = ap.parse_args()

    decode = {(r["model"], r["dataset"], r["transformation"]): r
              for r in read_rows(os.path.join(args.stats_dir, "decode_recovery_metrics.csv"))}
    summary = {(r["model"], r["dataset"], r["condition"]): r
               for r in read_rows(os.path.join(args.stats_dir, "summary_counts.csv"))}

    order = GRANULARITY_ORDER[:COARSE_N] if args.subset == "coarse" else GRANULARITY_ORDER
    keys = [k for k, _ in order]
    labels = [lbl for _, lbl in order]
    x = np.arange(len(keys))
    width = 0.40 if args.subset == "all" else 0.34

    fig_w = 14.0 if args.subset == "all" else 5.6
    fig_h = 5.4 if args.subset == "all" else 4.0
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(fig_w, fig_h), sharex=True)
    for ax, (ds, ds_label) in zip(np.atleast_1d(axes), DATASETS):
        solve, fidelity, strict, stub = [], [], [], []
        for t in keys:
            d = decode.get((args.model, ds, t))
            sm = summary.get((args.model, ds, t))
            # Graded reconstruction fidelity: word-level 4-gram F1 against the oracle
            # inverse. The binary gate below is the same comparison with an additional
            # verbatim-transcription requirement, so it is a floor on this.
            fidelity.append(100.0 * float(d["ngram4_f1"]) if d else np.nan)
            strict.append(100.0 * float(d["recovered_rate"]) if d else np.nan)
            stub.append(100.0 * float(d["stub_rate"]) if d else np.nan)
            solve.append(100.0 * float(sm["rate"]) if sm else np.nan)

        if args.show_fidelity:
            ax.bar(x - width / 2, fidelity, width, color=C_DECODE, alpha=0.45,
                   edgecolor="black", linewidth=0.6, zorder=2,
                   label="Graded reconstruction fidelity (4-gram F1)")
            ax.bar(x - width / 2, strict, width, color=C_DECODE, edgecolor="black",
                   linewidth=0.6, zorder=3, label="Verbatim recovery (CER $\\leq$ 0.02)")
        else:
            ax.bar(x - width / 2, strict, width, color=C_DECODE, edgecolor="black",
                   linewidth=0.6, zorder=3, label="Verbatim recovery (CER $\\leq$ 0.02)")
        ax.bar(x + width / 2, solve, width, facecolor="white", edgecolor=C_SOLVE,
               linewidth=1.6, label="Solve accuracy (transformed problem)", zorder=3)
        for xi, (st_, sv_) in enumerate(zip(strict, solve)):
            if np.isnan(st_) or np.isnan(sv_) or st_ - sv_ < 1.0 or sv_ < 5.0:
                continue
            ax.annotate("", xy=(xi - width / 2, st_ + 2), xytext=(xi + width / 2, sv_ + 2),
                        arrowprops=dict(arrowstyle="-|>", color="#2A7F3E", lw=1.5), zorder=6)
            ax.text(xi, max(st_, sv_) + 5, f"+{st_ - sv_:.0f}", ha="center", va="bottom",
                    fontsize=10.5 if args.subset == "coarse" else 13,
                    color="#2A7F3E", fontweight="bold", zorder=6)
        for xi, v in enumerate(stub):
            if args.subset == "coarse":
                break
            if not np.isnan(v) and v > STUB_THRESHOLD:
                ax.plot(xi - width / 2, 2.0, marker="v", markersize=8, color="#8172B3",
                        markeredgecolor="black", markeredgewidth=0.5, clip_on=False, zorder=5)
        if args.subset != "coarse":
            ax.plot([], [], linestyle="none", marker="v", markersize=8, color="#8172B3",
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=f"Stub rate > {STUB_THRESHOLD:.0f}% (model declines to attempt)")

        ax.set_title(ds_label, fontsize=13 if args.subset == "coarse" else 16,
                     fontweight="bold", pad=3, loc="left")
        ax.set_ylim(0, 118 if args.subset == "coarse" else 108)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
        ax.tick_params(axis="y", labelsize=10 if args.subset == "coarse" else 13)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    bottom = np.atleast_1d(axes)[-1]
    bottom.set_xticks(x)
    bottom.set_xticklabels(labels, fontsize=10.5 if args.subset == "coarse" else 13)
    bottom.set_xlabel(
        "Transformation, ordered by granularity (coarse $\\rightarrow$ fine)"
        if args.subset == "all" else "Transformations at word granularity and coarser",
        fontsize=11.5 if args.subset == "coarse" else 15)
    handles, lbls = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=1 if args.subset == "coarse" else 2,
               fontsize=9.5 if args.subset == "coarse" else 12.5, frameon=False)
    # Left inset in the rect reserves room for the shared y label below.
    plt.tight_layout(rect=[0.055 if args.subset == "coarse" else 0.02, 0, 1,
                           0.86 if args.subset == "coarse" else 0.90], h_pad=0.6)
    # One shared y label for both panels. The two bars are different metrics -- a
    # reconstruction rate and solve accuracy -- so the axis stays neutral.
    fig.supylabel("Percent", fontsize=12 if args.subset == "coarse" else 15, x=0.012)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.name}.pdf")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
