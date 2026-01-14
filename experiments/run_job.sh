#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
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
python eval_multiple.py --names context_saturation,interleaved_context,not_not_yot,opposites,sentence_reversal,word_reversal,word_split_swap,wrappers  --n_samples 5 --limit 30 --model "GAIR/LIMO-v2" > output_with_prompt_eng.txt
