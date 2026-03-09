#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=eval_out.out
#SBATCH --error=eval_out.out
#SBATCH --account=aip-gpekhime

# Run Evaluation
echo "Starting Evaluation..."

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
mkdir -p $HF_HOME
# NCCL Fixes
export NCCL_IGNORE_DISABLED_P2P=1
# export NCCL_DEBUG=INFO
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# python evaluate.py --names baseline --model Qwen/Qwen3-30B-A3B-Thinking-2507 --dataset HuggingFaceH4/aime_2024 --n_samples 16 --num_gpus 2 &> eval_out.out

