#!/bin/bash
# accuracy plots
python plot_results.py --aggregate --by_model --failures_on_top
python plot_results.py --aggregate --by_model --accuracy_overlay --recovery
python plot_results.py --aggregate --by_model --length
python plot_results.py --aggregate --single_metric