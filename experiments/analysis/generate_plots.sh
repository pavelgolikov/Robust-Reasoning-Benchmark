#!/bin/bash

# accuracy plots
python plot_results.py --model all --by_model --aggregate

# output length plots
python plot_results.py --model all --by_model --aggregate --length

# prompt recovery plots
python plot_results.py --model all --by_model --aggregate --recovery
