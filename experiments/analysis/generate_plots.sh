#!/bin/bash
# accuracy plots
python plot_results.py --aggregate --by_model --failures_on_top
python plot_results.py --aggregate --accuracy_overlay --recovery
python plot_results.py --aggregate --by_model --length
python plot_results.py --aggregate --single_metric
python plot_radar_categories.py
python plot_conditional_accuracy.py
python plot_global_conditional_accuracy.py
# python plot_over_under_thinking.py