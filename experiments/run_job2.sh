#!/bin/bash
#SBATCH --job-name=golikovp_job2
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
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

# TODO: need to run rail_fence eval real quick
# Running 5 samples per problem for averaging
# python evaluate.py --names rail_fence --n_samples 5 --num_gpu 4 --limit 30 > eval_out2.out
# python evaluate.py --names reversal --n_samples 5 --limit 30 --model "tiiuae/Falcon-H1R-7B" > eval_out.out

# python evaluate_conversation.py --max_model_length 262000 --n_samples 5 --num_distractors 15 --distractors_per_query 5 --num_gpus 4 > eval_out2.out

# python evaluate_conversation.py --max_model_length 262000 --n_samples 30 --context_fill_lvl 50 --distractors_per_query 4 --num_gpus 4 --new_run > eval_out2.out

# python evaluate_context.py --limit 2 --n_samples 3 --context_size 8192 --context_type math --context_file context_math_1M.json --num_gpus 2 &> eval_out2.out

python evaluate_agent.py --n_samples 5 --limit 1 --names rail_fence --num_gpus 2 &> eval_out2.out