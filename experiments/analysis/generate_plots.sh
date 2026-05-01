#!/bin/bash
# Consolidated and Optimized Plotting script
python3 visualize.py --plot_type accuracy --dataset MathArena/aime_2025
python3 visualize.py --plot_type output_length --dataset MathArena/aime_2025
python3 visualize.py --plot_type average_accuracy_drop --dataset MathArena/aime_2025
python3 visualize.py --plot_type radar_categories --dataset MathArena/aime_2025
