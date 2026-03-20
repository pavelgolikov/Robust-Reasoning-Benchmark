#!/bin/bash
#SBATCH --job-name=golikovp_job2
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=eval_out2.out
#SBATCH --error=eval_out2.out
#SBATCH --account=aip-gpekhime

# Run Evaluation
echo "Starting Evaluation..."

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
mkdir -p $HF_HOME

export NCCL_IGNORE_DISABLED_P2P=1

# python evaluate.py --names compound --num_distractors 2 --model mistralai/Ministral-3-14B-Reasoning-2512 --max_model_length 131072 --n_samples 16 --num_gpus 4 &> eval_out2.out

python evaluate.py --names baseline --model mistralai/Ministral-3-14B-Reasoning-2512 --max_model_length 131072 --n_samples 16 --num_gpus 4 &> eval_out2.out
