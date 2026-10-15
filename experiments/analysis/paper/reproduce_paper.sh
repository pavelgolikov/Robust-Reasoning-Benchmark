#!/bin/bash
# Regenerate every statistic, figure, and table row in the paper from the stored raw results.
# CPU only: no model is queried. Takes about 7 minutes (step 2 dominates).
#
#   bash experiments/analysis/paper/reproduce_paper.sh               # regenerate everything
#   bash experiments/analysis/paper/reproduce_paper.sh /path/to/paper # ...and copy figures to <paper>/plots/
#
# The paper's own Makefile crops the copied PDFs into plots/*-crop.pdf.
set -euo pipefail
cd "$(dirname "$0")/.."            # experiments/analysis

echo "== 1/4  statistics -> rebuttal_stats/summary_counts.csv, paired_comparisons.csv"
python3 rebuttal_statistics.py --out_dir rebuttal_stats

echo "== 2/4  decode-only recovery -> rebuttal_stats/decode_recovery_metrics.csv"
python3 decode_recovery_metrics.py --out_dir rebuttal_stats

echo "== 3/4  figures"
python3 visualize_combined.py --plot_type compound --compact                          # Figure 1  -> plots/compound.pdf
python3 visualize_combined.py --plot_type accuracy --compact                          # Figure 3  -> plots/accuracy.pdf
python3 visualize_combined.py --plot_type average_accuracy_drop                       # Figure 4  -> plots/average_accuracy_drop.pdf
python3 plot_decode_recovery_paper.py --subset coarse --name decode_recovery          # Figure 5  -> plots/decode_recovery.pdf
python3 visualize_combined.py --plot_type output_length                               # Figure 7  -> plots/length.pdf
python3 plot_decode_recovery_paper.py --subset all --show_fidelity \
        --name decode_recovery_full                                                   # Figure 11 -> plots/decode_recovery_full.pdf
( cd ../compound/attentions && \
  RRB_ATTN_ROW_H=1.9 RRB_ATTN_HSPACE=0.35 python3 plot_attention_heatmaps.py )       # Figure 6  -> compound/attentions/combined_attention_dilution.pdf

echo "== 4/4  in-text numbers and LaTeX rows of Tables 2-5"
python3 paper/paper_numbers.py

if [ $# -ge 1 ]; then
    dest="$1/plots"
    cp plots/compound.pdf plots/accuracy.pdf plots/average_accuracy_drop.pdf plots/length.pdf \
       plots/decode_recovery.pdf plots/decode_recovery_full.pdf "$dest/"
    cp ../compound/attentions/combined_attention_dilution.pdf "$dest/"
    echo "copied figures to $dest"
fi
