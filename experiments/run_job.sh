#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
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

# NCCL Fixes
# export NCCL_DEBUG=INFO
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# python evaluate.py --names rail_fence --n_samples 5 --limit 30 > eval_out.out
# python evaluate.py --names reversal --n_samples 5 --limit 30 --model "tiiuae/Falcon-H1R-7B" > eval_out.out
# python evaluate_agent.py --max_model_length 65536 --limit 30 --n_samples 5 --names interleaved_context_word,not_not --num_gpus 4 > eval_out.out

python evaluate_conversation.py --max_model_length 262000 --sample_range 0 --n_samples 5 --num_distractors 16 --distractors_per_query 4 --num_gpus 4 > eval_out.out

# last launched 
# python evaluate_conversation.py --max_model_length 262000 --sample_range 0-10 --n_samples 5 --num_distractors 16 --distractors_per_query 4 --num_gpus 4 &> eval_out.out
