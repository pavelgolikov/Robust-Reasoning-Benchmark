
#!/bin/bash
#SBATCH --job-name=golikovp_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
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


# perturb eval
# python evaluate.py --names all --temperature 0.7 --top_p 1.0 --dataset MathArena/aime_2025 --model Qwen/Qwen3-30B-A3B-Thinking-2507          --max_model_length 32000  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names rail_fence,snake_vertical,snake_horizontal,rectangle_perimeter  --temperature 0.7 --top_p 1.0 --dataset MathArena/aime_2025 --model nvidia/OpenReasoning-Nemotron-7B          --max_model_length 32000  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names all --temperature 0.7 --top_p 1.0 --dataset MathArena/aime_2025 --model openai/gpt-oss-120b                       --max_model_length 32000  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names snake_horizontal,rectangle_perimeter --temperature 0.7 --top_p 1.0 --dataset MathArena/aime_2025 --model nvidia/OpenReasoning-Nemotron-32B         --max_model_length 32000  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names snake_vertical,snake_horizontal,rectangle_perimeter --temperature 0.7 --top_p 1.0 --dataset MathArena/aime_2025 --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --max_model_length 32000  --n_samples 16 --num_gpus 4 &> eval_out.out

# compound eval on AIME 2025
# python evaluate.py --dataset MathArena/aime_2025 --names baseline                     --model Qwen/Qwen3-30B-A3B-Thinking-2507          --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --dataset MathArena/aime_2025 --names compound --num_distractors 3 --model Qwen/Qwen3-30B-A3B-Thinking-2507          --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --dataset MathArena/aime_2025 --names baseline                     --model nvidia/OpenReasoning-Nemotron-7B          --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --dataset MathArena/aime_2025 --names compound --num_distractors 3 --model nvidia/OpenReasoning-Nemotron-7B          --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --dataset MathArena/aime_2025 --names baseline                     --model openai/gpt-oss-120b                       --max_model_length 131072 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --dataset MathArena/aime_2025 --names compound --num_distractors 3 --model openai/gpt-oss-120b                       --max_model_length 131072 --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --dataset MathArena/aime_2025 --names baseline                     --model nvidia/OpenReasoning-Nemotron-32B         --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --dataset MathArena/aime_2025 --names compound --num_distractors 3 --model nvidia/OpenReasoning-Nemotron-32B         --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --dataset MathArena/aime_2025 --names baseline                     --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --dataset MathArena/aime_2025 --names compound --num_distractors 3 --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out

# compound eval on AIME 2024

# python evaluate.py --names baseline                     --model Qwen/Qwen3-30B-A3B-Thinking-2507          --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names compound --num_distractors 3 --model Qwen/Qwen3-30B-A3B-Thinking-2507          --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --names baseline                     --model nvidia/OpenReasoning-Nemotron-7B          --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names compound --num_distractors 3 --model nvidia/OpenReasoning-Nemotron-7B          --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --names baseline                     --model openai/gpt-oss-120b                       --max_model_length 131072 --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names compound --num_distractors 2 --model openai/gpt-oss-120b                       --max_model_length 131072 --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --names baseline                     --model nvidia/OpenReasoning-Nemotron-32B         --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate.py --names compound --num_distractors 3 --model nvidia/OpenReasoning-Nemotron-32B         --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out

# python evaluate.py --names baseline                     --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out
python evaluate.py --names compound --num_distractors 3 --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --max_model_length 131072  --n_samples 16 --num_gpus 4 &> eval_out.out

# context eval
# python evaluate_context.py --model nvidia/OpenReasoning-Nemotron-7B --context_size 98304 --context_type math --context_file /home/golikovp/projects/aip-gpekhime/golikovp/Linguistic_traps/experiments/context_saturation/contexts/context_math_98304_Nemotron-7B.json --max_model_len 32000  --n_samples 16 --num_gpus 4 &> eval_out.out
# python evaluate_context.py --model nvidia/OpenReasoning-Nemotron-32B --context_size 98304 --context_type math --context_file /home/golikovp/projects/aip-gpekhime/golikovp/Linguistic_traps/experiments/context_saturation/contexts/context_math_98304_Nemotron-32B.json --n_samples 16 --num_gpus 4 &> eval_out.out

