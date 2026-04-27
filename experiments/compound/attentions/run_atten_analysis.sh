#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
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
python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/openai_gpt-oss-120b/MathArena_aime_2025/openai_gpt-oss-120b_MathArena_aime_2025_compound_s42_20260405_153738.json &> atten_out_oss.out
# python analyze_dilution.py --json_file /home/golikovp/projects/aip-gpekhime/golikovp/Robust-Reasoning-Benchmark/experiments/compound/results/nvidia_OpenReasoning-Nemotron-7B/MathArena_aime_2025/nvidia_OpenReasoning-Nemotron-7B_MathArena_aime_2025_compound_s42_20260405_054949.json &> atten_out_nemo.out

