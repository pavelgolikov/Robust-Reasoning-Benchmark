#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
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
# python evaluate.py --name interleaved_context --n_samples 5 --limit 30 --model "GAIR/LIMO-v2"
# python evaluate.py --names all --n_samples 5 --limit 30 --model "GAIR/LIMO-v2" > slurm_out.out
python3 analysis/analyze_failures.py --file interleaved_context_word/results/GAIR_LIMO-v2_interleaved_context_word_s42_20260118_040052.json --model_name all-MiniLM-L6-v2
