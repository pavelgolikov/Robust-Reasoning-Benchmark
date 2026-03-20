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

# output lengths by model:
# Qwen/Qwen3-30B-A3B-Thinking-2507: 81920
# mistralai/Ministral-3-14B-Reasoning-2512


# openai/gpt-oss-120b: 131072
# tiiuae/Falcon-H1R-7B: 65536
# deepseek-ai/DeepSeek-R1-Distill-Llama-70B: 32768
# GAIR/LIMO-v2: 8k - consider replacing

# python evaluate.py --names interleaved_context_word,interleaved_context_symbol,rail_fence,snake_vertical,snake_horizontal,rectangle_perimeter --model Qwen/Qwen3-30B-A3B-Thinking-2507 --dataset MathArena/aime_2025 --n_samples 16 --num_gpus 4 &> eval_out.out


# python evaluate.py --names compound --num_distractors 2 --model openai/gpt-oss-120b                       --max_model_length 131072 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names compound --num_distractors 2 --model Qwen/Qwen3-30B-A3B-Thinking-2507          --max_model_length 81920  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names compound --num_distractors 1 --model tiiuae/Falcon-H1R-7B                      --max_model_length 65536  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names compound --num_distractors 1 --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --max_model_length 32768  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names compound --num_distractors 1 --model GAIR/LIMO-v2                              --max_model_length 8192   --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --names baseline --model openai/gpt-oss-120b                       --max_model_length 131072 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names baseline --model Qwen/Qwen3-30B-A3B-Thinking-2507          --max_model_length 81920  --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --names baseline                     --model tiiuae/Falcon-H1R-7B                      --max_model_length 65536  --n_samples 16 --num_gpus 2 &> eval_out.out
python evaluate.py --names compound --num_distractors 1 --model tiiuae/Falcon-H1R-7B                      --max_model_length 65536  --n_samples 16 --num_gpus 2 &> eval_out.out



# python evaluate.py --names all --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --dataset MathArena/aime_2025 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names all --model GAIR/LIMO-v2 --dataset MathArena/aime_2025 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names baseline,not_not,rectangle_perimeter,snake_horizontal,snake_vertical --model tiiuae/Falcon-H1R-7B --dataset MathArena/aime_2025 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names rectangle_perimeter --model Qwen/Qwen3-30B-A3B-Thinking-2507 --dataset HuggingFaceH4/aime_2024 --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --names rectangle_perimeter --model openai/gpt-oss-120b --dataset MathArena/aime_2025 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names all --model  --dataset MathArena/aime_2025 --n_samples 16 --num_gpus 4 &> eval_out.out

# python analysis/prompt_reconstruction/analyze_prompt_recovery_llm.py --names all --model all --num_gpus 4 &> eval_out.out

# python evaluate.py --names rectangle_perimeter,snake_horizontal,snake_vertical --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --dataset HuggingFaceH4/aime_2024 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names snake_vertical --model GAIR/LIMO-v2 --dataset HuggingFaceH4/aime_2024 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names snake_horizontal,snake_vertical --model tiiuae/Falcon-H1R-7B --dataset HuggingFaceH4/aime_2024 --n_samples 16 --num_gpus 4 &> eval_out.out

# python3 analysis/prompt_reconstruction/analyze_prompt_recovery.py --model all --dataset HuggingFaceH4/aime_2024 --names rectangle_perimeter,snake_vertical,snake_horizontal --skip_existing

# python analysis/prompt_reconstruction/analyze_prompt_recovery.py --model all
