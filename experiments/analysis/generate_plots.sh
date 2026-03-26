#!/bin/bash
# Consolidated and Optimized Plotting script
python3 visualize.py --plot_type accuracy
python3 visualize.py --plot_type output_length
python3 visualize.py --plot_type average_accuracy_drop
python3 visualize.py --plot_type radar_categories
