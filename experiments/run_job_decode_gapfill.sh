#!/bin/bash
#SBATCH --job-name=golikovp_decode_gapfill
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --output=decode_gapfill_out.out
#SBATCH --error=decode_gapfill_out.out
#SBATCH --account=aip-gpekhime

# The three decode-only recovery cells missing from decode_recovery/results, lost when the
# earlier job hit its 7h wall clock (see eval_out.out). Same settings as the completed runs.

export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
mkdir -p $HF_HOME
export NCCL_IGNORE_DISABLED_P2P=1

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0
module load arrow/25.0.0

COMMON="--max_model_length 32000 --max_tokens 32000 --n_samples 16 --num_gpus 4 --temperature 0.7 --top_p 1.0"

python evaluate_decode_recovery.py --dataset HuggingFaceH4/aime_2024 --model nvidia/OpenReasoning-Nemotron-32B --names snake_horizontal            $COMMON &> decode_recovery_gapfill_nemotron32_aime2024.out
python evaluate_decode_recovery.py --dataset MathArena/aime_2025     --model nvidia/OpenReasoning-Nemotron-7B  --names rail_fence,split_reversal   $COMMON &> decode_recovery_gapfill_nemotron7_aime2025.out
