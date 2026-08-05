#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python experiments/analysis/rebuttal_statistics.py "$@"
