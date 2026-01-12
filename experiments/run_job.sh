#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_out.out
#SBATCH --error=slurm_out.err
#SBATCH --account=aip-gpekhime

# Run Evaluation
echo "Starting Evaluation..."

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
mkdir -p $HF_HOME

# Running 5 samples per problem for averaging
# python evaluate.py --name sentence_reversal --n_samples 5 --limit 30 --model "GAIR/LIMO-v2"
# python evaluate.py --name word_reversal --n_samples 5 --limit 30 --model "GAIR/LIMO-v2"
python evaluate.py --name word_split_swap --n_samples 5 --limit 30 --model "GAIR/LIMO-v2"
