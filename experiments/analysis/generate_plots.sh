#!/bin/bash

# accuracy plots
python plot_results.py --model all --by_model --aggregate

# output length plots
python plot_results.py --model all --by_model --aggregate --length

# prompt recovery plots
python plot_results.py --model all --by_model --aggregate --recovery

python plot_results.py --by_model --aggregate --accuracy_overlay --recovery

python plot_results.py --by_model --aggregate --failures_on_top

# single metric (average drop) plot
python plot_results.py --aggregate --single_metric