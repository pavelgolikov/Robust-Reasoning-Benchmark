#!/bin/bash
#SBATCH --job-name=golikovp_job2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
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

# python evaluate.py --names baseline --dataset HuggingFaceH4/aime_2024 --n_samples 16 --num_gpus 2 &> eval_out2.out

# python evaluate_context.py \
#   --model openai/gpt-oss-120b \
#   --context_file /home/golikovp/projects/aip-gpekhime/golikovp/Linguistic_traps/experiments/context_saturation/contexts/context_math_98304_openai_gpt-oss-120b.json \
#   --context_type math \
#   --max_model_len 128000 \
#   --context_size 98304 \
#   --max_tokens 32000 \
#   --n_samples 8 \
#   --num_gpus 4 &>> eval_out2.out

