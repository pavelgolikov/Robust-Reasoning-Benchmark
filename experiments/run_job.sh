#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=eval_out.out
#SBATCH --error=eval_err.err
#SBATCH --account=aip-gpekhime

# Run Evaluation
echo "Starting Evaluation..."

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
mkdir -p $HF_HOME

# Running 5 samples per problem for averaging
python evaluate.py --names reversal --n_samples 5 --limit 30 --model "tiiuae/Falcon-H1R-7B" > eval_out.out
# python evaluate.py --names reversal --n_samples 5 --limit 30 --model "tiiuae/Falcon-H1R-7B" > eval_out.out
