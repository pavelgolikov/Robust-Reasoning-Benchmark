#!/bin/bash
#SBATCH --job-name=golikovp_job_reb
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --output=eval_out.out
#SBATCH --error=eval_out.out
#SBATCH --account=aip-gpekhime

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
mkdir -p $HF_HOME
export NCCL_IGNORE_DISABLED_P2P=1
# NCCL Fixes

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0
module load arrow/25.0.0

# decode recovery eval on AIME 2024
# python evaluate_decode_recovery.py --dataset HuggingFaceH4/aime_2024 --names all --model Qwen/Qwen3-30B-A3B-Thinking-2507          --max_model_length 32000  --max_tokens 32000 --n_samples 16 --num_gpus 4 --temperature 0.7 --top_p 1.0 &> decode_recovery_qwen_aime2024.out
python evaluate_decode_recovery.py --dataset HuggingFaceH4/aime_2024 --names all --model nvidia/OpenReasoning-Nemotron-32B         --max_model_length 32000  --max_tokens 32000 --n_samples 16 --num_gpus 4 --temperature 0.7 --top_p 1.0 &> decode_recovery_nemotron32_aime2024.out
python evaluate_decode_recovery.py --dataset HuggingFaceH4/aime_2024 --names all --model nvidia/OpenReasoning-Nemotron-7B          --max_model_length 32000  --max_tokens 32000 --n_samples 16 --num_gpus 4 --temperature 0.7 --top_p 1.0 &> decode_recovery_nemotron7_aime2024.out

# decode recovery eval on AIME 2025
python evaluate_decode_recovery.py --dataset MathArena/aime_2025 --names all --model Qwen/Qwen3-30B-A3B-Thinking-2507              --max_model_length 32000  --max_tokens 32000 --n_samples 16 --num_gpus 4 --temperature 0.7 --top_p 1.0 &> decode_recovery_qwen_aime2025.out
python evaluate_decode_recovery.py --dataset MathArena/aime_2025 --names all --model nvidia/OpenReasoning-Nemotron-32B             --max_model_length 32000  --max_tokens 32000 --n_samples 16 --num_gpus 4 --temperature 0.7 --top_p 1.0 &> decode_recovery_nemotron32_aime2025.out
python evaluate_decode_recovery.py --dataset MathArena/aime_2025 --names all --model nvidia/OpenReasoning-Nemotron-7B              --max_model_length 32000  --max_tokens 32000 --n_samples 16 --num_gpus 4 --temperature 0.7 --top_p 1.0 &> decode_recovery_nemotron7_aime2025.out

# # passive context eval on AIME 2024
# python evaluate_passive_context.py --dataset HuggingFaceH4/aime_2024 --model Qwen/Qwen3-30B-A3B-Thinking-2507          --max_model_length 131072  --max_tokens 131072 --n_samples 16 --num_distractors 3 --num_gpus 4 --temperature 0.6 --top_p 0.95 &> passive_context_qwen_aime2024.out
# python evaluate_passive_context.py --dataset HuggingFaceH4/aime_2024 --model nvidia/OpenReasoning-Nemotron-32B         --max_model_length 131072  --max_tokens 131072 --n_samples 16 --num_distractors 3 --num_gpus 4 --temperature 0.6 --top_p 0.95 &> passive_context_nemotron32_aime2024.out
# python evaluate_passive_context.py --dataset HuggingFaceH4/aime_2024 --model nvidia/OpenReasoning-Nemotron-7B          --max_model_length 131072  --max_tokens 131072 --n_samples 16 --num_distractors 3 --num_gpus 4 --temperature 0.6 --top_p 0.95 &> passive_context_nemotron7_aime2024.out

# # passive context eval on AIME 2025
# python evaluate_passive_context.py --dataset MathArena/aime_2025 --model Qwen/Qwen3-30B-A3B-Thinking-2507              --max_model_length 131072  --max_tokens 131072 --n_samples 16 --num_distractors 3 --num_gpus 4 --temperature 0.6 --top_p 0.95 &> passive_context_qwen_aime2025.out
# python evaluate_passive_context.py --dataset MathArena/aime_2025 --model nvidia/OpenReasoning-Nemotron-32B             --max_model_length 131072  --max_tokens 131072 --n_samples 16 --num_distractors 3 --num_gpus 4 --temperature 0.6 --top_p 0.95 &> passive_context_nemotron32_aime2025.out
# python evaluate_passive_context.py --dataset MathArena/aime_2025 --model nvidia/OpenReasoning-Nemotron-7B              --max_model_length 131072  --max_tokens 131072 --n_samples 16 --num_distractors 3 --num_gpus 4 --temperature 0.6 --top_p 0.95 &> passive_context_nemotron7_aime2025.out
