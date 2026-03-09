#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=7:00:00
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

python evaluate.py --names all --model Qwen/Qwen3-30B-A3B-Thinking-2507 --dataset HuggingFaceH4/aime_2024 --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate_agent.py --names wrappers,split_reversal,sentence_reversal,word_reversal,rail_fence --model GAIR/LIMO-v2 --dataset HuggingFaceH4/aime_2024 --max_model_length 32000 --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate_context.py --n_samples 5 --context_size 16384 --context_type math --context_file context_math_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 16384 --context_type text --context_file context_text_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 32768 --context_type math --context_file context_math_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 32768 --context_type text --context_file context_text_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 49152 --context_type math --context_file context_math_1M.json --num_gpus 4 &> eval_out.out
# python evaluate_context.py --n_samples 5 --context_size 49152 --context_type text --context_file context_text_1M.json --num_gpus 4 &> eval_out.out
