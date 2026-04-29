#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=7:00:00
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

# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/deepseek-ai_DeepSeek-R1-Distill-Llama-70B/MathArena_aime_2025/deepseek-ai_DeepSeek-R1-Distill-Llama-70B_MathArena_aime_2025_compound_s42_20260407_135124.json > atten_out_dsr.out
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/openai_gpt-oss-120b/MathArena_aime_2025/openai_gpt-oss-120b_MathArena_aime_2025_compound_s42_20260405_153738.json

# Qwen 30B
# 1 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/Qwen_Qwen3-30B-A3B-Thinking-2507/MathArena_aime_2025/Qwen_Qwen3-30B-A3B-Thinking-2507_MathArena_aime_2025_compound_s42_20260330_155901.json
# 2 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/Qwen_Qwen3-30B-A3B-Thinking-2507/MathArena_aime_2025/Qwen_Qwen3-30B-A3B-Thinking-2507_MathArena_aime_2025_compound_s42_20260403_055159.json
# 3 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/Qwen_Qwen3-30B-A3B-Thinking-2507/MathArena_aime_2025/Qwen_Qwen3-30B-A3B-Thinking-2507_MathArena_aime_2025_compound_s42_20260405_035834.json

# Nemotron 7B
# 1 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/nvidia_OpenReasoning-Nemotron-7B/MathArena_aime_2025/nvidia_OpenReasoning-Nemotron-7B_MathArena_aime_2025_compound_s42_20260330_172942.json
# 2 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/nvidia_OpenReasoning-Nemotron-7B/MathArena_aime_2025/nvidia_OpenReasoning-Nemotron-7B_MathArena_aime_2025_compound_s42_20260403_071356.json
# 3 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/nvidia_OpenReasoning-Nemotron-7B/MathArena_aime_2025/nvidia_OpenReasoning-Nemotron-7B_MathArena_aime_2025_compound_s42_20260405_054949.json

# DeepSeek R1 Distill Llama 70B
# 1 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/deepseek-ai_DeepSeek-R1-Distill-Llama-70B/MathArena_aime_2025/deepseek-ai_DeepSeek-R1-Distill-Llama-70B_MathArena_aime_2025_compound_s42_20260402_002724.json
# 2 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/deepseek-ai_DeepSeek-R1-Distill-Llama-70B/MathArena_aime_2025/deepseek-ai_DeepSeek-R1-Distill-Llama-70B_MathArena_aime_2025_compound_s42_20260404_164148.json
# 3 distractor
python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/deepseek-ai_DeepSeek-R1-Distill-Llama-70B/MathArena_aime_2025/deepseek-ai_DeepSeek-R1-Distill-Llama-70B_MathArena_aime_2025_compound_s42_20260407_135124.json

# Nemotron 32B
# 1 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/nvidia_OpenReasoning-Nemotron-32B/MathArena_aime_2025/nvidia_OpenReasoning-Nemotron-32B_MathArena_aime_2025_compound_s42_20260330_231545.json
# 2 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/nvidia_OpenReasoning-Nemotron-32B/MathArena_aime_2025/nvidia_OpenReasoning-Nemotron-32B_MathArena_aime_2025_compound_s42_20260404_015745.json
# 3 distractor
python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/nvidia_OpenReasoning-Nemotron-32B/MathArena_aime_2025/nvidia_OpenReasoning-Nemotron-32B_MathArena_aime_2025_compound_s42_20260406_102432.json

# GPT-OSS 120B
# 1 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/openai_gpt-oss-120b/MathArena_aime_2025/openai_gpt-oss-120b_MathArena_aime_2025_compound_s42_20260401_115741.json
# 2 distractor
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/openai_gpt-oss-120b/MathArena_aime_2025/openai_gpt-oss-120b_MathArena_aime_2025_compound_s42_20260404_075545.json
# 3 distractor
python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/openai_gpt-oss-120b/MathArena_aime_2025/openai_gpt-oss-120b_MathArena_aime_2025_compound_s42_20260405_153738.json

