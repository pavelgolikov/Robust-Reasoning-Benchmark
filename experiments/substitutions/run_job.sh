#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4      # CPU cores per task
#SBATCH --mem=32G              # Memory
#SBATCH --time=04:00:00        # Max run time (4 hours)
#SBATCH --output=slurm_out.out
#SBATCH --error=slurm_out.err
#SBATCH --account=aip-gpekhime

# Run Evaluation
echo "Starting Evaluation..."

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
mkdir -p $HF_HOME

# Run the consolidated baseline evaluation
# Default model is GAIR/LIMO, but can be overridden with --model if needed
# Running 5 samples per problem for objective averaging
python evaluate.py --n_samples 5 --limit 30
