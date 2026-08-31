#!/bin/bash
#SBATCH --job-name=golikovp_passive_unique
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=23:00:00
#SBATCH --output=passive_unique_out.out
#SBATCH --error=passive_unique_out.out
#SBATCH --account=aip-gpekhime

# Equal-length passive-context control, re-run with NON-REPEATING text.
#
# The original runs (passive_context/results/*_passive_context_d3_*.json) padded the
# context by repeating a 411-word Gibbon excerpt ~90-125 times, which is a degenerate
# repetition loop rather than neutral filler. These runs draw one contiguous window from
# gibbon_decline_and_fall_long.txt instead; --passive_mode unique makes the script fail
# loudly rather than silently repeat if the corpus is ever too short.

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

# Preflight, so a bad corpus or a stale checkout fails in seconds instead of after loading a
# 30B model: check that the passive corpus is long enough under each tokenizer, and record the
# position-embedding configuration in the log alongside the results it explains.
python - <<'PREFLIGHT' || exit 1
from transformers import AutoConfig, AutoTokenizer

REQUIRED_TOKENS = 100_000  # the longest context any run has needed so far is ~69k
corpus = open("passive_context/gibbon_decline_and_fall_long.txt", encoding="utf-8").read()

ok = True
for model in [
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "nvidia/OpenReasoning-Nemotron-32B",
    "nvidia/OpenReasoning-Nemotron-7B",
]:
    config = AutoConfig.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    n = len(tokenizer.encode(corpus, add_special_tokens=False))
    print(
        f"{model}: corpus={n} tokens | max_position_embeddings="
        f"{config.max_position_embeddings} | rope_scaling={getattr(config, 'rope_scaling', None)}"
    )
    if n < REQUIRED_TOKENS:
        print(f"  FAIL: corpus has {n} tokens, need at least {REQUIRED_TOKENS}")
        ok = False

raise SystemExit(0 if ok else 1)
PREFLIGHT

COMMON="--passive_mode unique --num_distractors 3 --max_model_length 131072 --max_tokens 131072 --n_samples 16 --num_gpus 4 --temperature 0.6 --top_p 0.95"

# AIME 2024
python evaluate_passive_context.py --dataset HuggingFaceH4/aime_2024 --model Qwen/Qwen3-30B-A3B-Thinking-2507  $COMMON &> passive_unique_qwen_aime2024.out
python evaluate_passive_context.py --dataset HuggingFaceH4/aime_2024 --model nvidia/OpenReasoning-Nemotron-32B $COMMON &> passive_unique_nemotron32_aime2024.out
python evaluate_passive_context.py --dataset HuggingFaceH4/aime_2024 --model nvidia/OpenReasoning-Nemotron-7B  $COMMON &> passive_unique_nemotron7_aime2024.out

# AIME 2025
python evaluate_passive_context.py --dataset MathArena/aime_2025 --model Qwen/Qwen3-30B-A3B-Thinking-2507      $COMMON &> passive_unique_qwen_aime2025.out
python evaluate_passive_context.py --dataset MathArena/aime_2025 --model nvidia/OpenReasoning-Nemotron-32B     $COMMON &> passive_unique_nemotron32_aime2025.out
python evaluate_passive_context.py --dataset MathArena/aime_2025 --model nvidia/OpenReasoning-Nemotron-7B      $COMMON &> passive_unique_nemotron7_aime2025.out
