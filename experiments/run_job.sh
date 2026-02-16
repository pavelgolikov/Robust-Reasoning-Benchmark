#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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

python evaluate.py --names rail_fence --n_samples 5 --num_gpus 2 &> eval_out.out
# python evaluate_agent.py --max_model_length 65536 --n_samples 5 --names rail_fence --num_gpus 4 &> eval_out.out
# python evaluate_context.py --limit 2 --n_samples 3 --context_size 8192 --context_type math --context_file context_math_1M.json --num_gpus 2 &> eval_out.out

# python evaluate.py --names reversal --n_samples 5 --limit 30 --model "tiiuae/Falcon-H1R-7B" > eval_out.out
# python evaluate_agent.py --max_model_length 65536 --limit 30 --n_samples 5 --names interleaved_context_word,not_not --num_gpus 4 > eval_out.out

# python evaluate_conversation.py \
#     --dataset MathArena/aime_2025 \
#     --split train \
#     --max_model_length 65536 \
#     --max_saturation_step_tokens 8192 \
#     --n_samples 5 \
#     --context_saturation 25 \
#     --distractors_per_query 4 \
#     --num_gpus 4 \
#     > eval_out.out

# python evaluate_context.py --n_samples 5 --context_size 16384 --context_type math --context_file context_math_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 16384 --context_type text --context_file context_text_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 32768 --context_type math --context_file context_math_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 32768 --context_type text --context_file context_text_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 49152 --context_type math --context_file context_math_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 49152 --context_type text --context_file context_text_1M.json --num_gpus 4 &> eval_out.out
