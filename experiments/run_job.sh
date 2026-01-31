#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
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

# TODO: need to run rail_fence eval real quick
# Running 5 samples per problem for averaging
# python evaluate.py --names rail_fence --n_samples 5 --limit 30 > eval_out.out
# python evaluate.py --names reversal --n_samples 5 --limit 30 --model "tiiuae/Falcon-H1R-7B" > eval_out.out

python evaluate_conversation.py --max_model_length 262000 --batch_size 150 --n_samples 5 --num_distractors 30 --num_gpus 4 > eval_out.out
