#!/usr/bin/env python3
import json
import argparse
import os
import re
import glob
import time
from collections import defaultdict
from typing import List, Dict, Any

from tqdm import tqdm

# ── Configuration (mirrored from other scripts) ────────────────────

TECHNIQUES_LIST = [
    "baseline",
    "not_not", "opposites", "wrappers",
    "interleaved_context_line", "interleaved_context_word", "interleaved_context_symbol",
    "context_saturation",
    "sentence_reversal", "word_reversal", "split_reversal",
    "rail_fence",
    "rectangle_perimeter", "snake_vertical", "snake_horizontal",
]

# ── File discovery ───────────────────────────────────────────────────

def find_latest_result(experiment_name: str, model_name: str, dataset_name: str, base_dir: str) -> str:
    """Finds the latest JSON result file for a given experiment/model/dataset."""
    safe_model = model_name.replace('/', '_').replace(' ', '_')
    safe_dataset = dataset_name.replace('/', '_')
    results_dir = os.path.join(base_dir, experiment_name, "results", safe_model, safe_dataset)

    if not os.path.exists(results_dir):
        return None

    files = glob.glob(os.path.join(results_dir, "*.json"))
    files = [f for f in files if not f.endswith("_prompt_recovery.json")
             and "semantic" not in f and not f.endswith("_raw.json")
             and "summary" not in f and not f.startswith("jobs_") 
             and not f.startswith("batch_") and not f.startswith("tracking_")]

    if not files:
        return None
    return max(files, key=os.path.getmtime)


def discover_models(base_dir: str, techniques: List[str], dataset_name: str) -> List[str]:
    """Auto-discover all model names that have results in any of the given techniques."""
    safe_dataset = dataset_name.replace('/', '_')
    models = set()
    for technique in techniques:
        results_dir = os.path.join(base_dir, technique, "results")
        if not os.path.isdir(results_dir):
            continue
        for model_dir_name in os.listdir(results_dir):
            dataset_dir = os.path.join(results_dir, model_dir_name, safe_dataset)
            if os.path.isdir(dataset_dir):
                json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
                json_files = [f for f in json_files if not f.endswith("_prompt_recovery.json")]
                if json_files:
                    models.add(model_dir_name)
    return sorted(models)

# ── Judge Prompt Generation ──────────────────────────────────────────

def build_judge_messages(target: str, output: str) -> List[Dict[str, str]]:
    sys_prompt = "You are an expert evaluator classifying an AI model's internal reasoning."
    user_prompt = f"""[ORIGINAL PROBLEM]
{target}

[MODEL RESPONSE TO DISGUISED VERSION]
{output}

Task: Did the model successfully recognize/decode the original math problem? 
Criteria for YES:
1. The model explicitly reconstructed the correct original problem.
2. OR the model's math steps clearly demonstrate it understood the true numbers, variables, and relationships of the original problem unharmed by the disguise.
Criteria for NO:
1. The model was confused by reversed/scrambled words and tried to solve the gibberish literally.
2. The model hallucinated a completely different problem.

Answer EXACTLY "YES" or "NO" with no other text."""
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Prompt Recovery using LLM-as-a-judge (vLLM)")
    parser.add_argument("--judge_model", type=str, default="openai/gpt-oss-120b", help="Path/HuggingFace ID for the judge LLM")
    parser.add_argument("--names", type=str, default='all', help="Comma-separated technique names or 'all'")
    parser.add_argument("--model", type=str, required=True, help="Target model name to evaluate (e.g. GAIR/LIMO-v2) or 'all' to auto-discover")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--skip_existing", action="store_true", help="Skip model/technique combos that already have a recovery JSON")
    parser.add_argument("--dry", action="store_true", help="Dry run: just collect prompts but don't load vLLM")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for the judge model")
    parser.add_argument("--max_model_length", type=int, default=None, help="Max context length for the judge (if None, infers from model config)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.dirname(os.path.dirname(script_dir))
    base_dir = experiments_dir

    safe_dataset = args.dataset.replace('/', '_')

    if args.names == 'all':
        techniques = TECHNIQUES_LIST
    else:
        techniques = [n.strip() for n in args.names.split(',') if n.strip()]

    if args.model == 'all':
        target_models = discover_models(base_dir, techniques, args.dataset)
        print(f"Auto-discovered {len(target_models)} target models: {target_models}")
    else:
        target_models = [args.model.replace('/', '_').replace(' ', '_')]

    if not target_models:
        print("No target models found.")
        return

    # 1. Collect all requests
    print("\nCollecting evaluation candidates...")
    
    # We will build a list of tasks. 
    # tasks = [ { "target_model": M, "technique": T, "latest_file": F, "entries": [...] }, ... ]
    all_experiments = []
    
    # We will build a flat list of items that require LLM grading.
    # llm_requests = [ { "exp_idx": int, "entry_idx": int, "messages": [...] }, ... ]
    llm_requests = []

    for model_name in target_models:
        output_dir = os.path.join(base_dir, "analysis", "prompt_reconstruction", "results", model_name, safe_dataset)
        os.makedirs(output_dir, exist_ok=True)

        for technique_name in techniques:
            if args.skip_existing:
                existing = glob.glob(os.path.join(output_dir, f"{technique_name}_prompt_recovery_*.json"))
                if existing:
                    continue

            latest_file = find_latest_result(technique_name, model_name, args.dataset, base_dir)
            if not latest_file:
                continue

            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "results" in data:
                data = data["results"]
            elif isinstance(data, dict):
                data = list(data.values())

            if not data:
                continue
            
            exp_idx = len(all_experiments)
            all_experiments.append({
                "target_model": model_name,
                "technique": technique_name,
                "latest_file": latest_file,
                "output_dir": output_dir,
                "entries": data,
                "summary": {
                    "source_file": latest_file,
                    "total_samples": len(data),
                    "original_correct": 0,
                    "semantic_correct": 0,
                    "recovered_cases": []
                }
            })

            # Prepare prompts for entries that need it
            for entry_idx, entry in enumerate(data):
                is_orig_correct = entry.get('correct', False)
                if is_orig_correct:
                    # Auto-override: Correct answer implies successful prompt decoding
                    all_experiments[exp_idx]["summary"]["original_correct"] += 1
                    all_experiments[exp_idx]["summary"]["semantic_correct"] += 1
                    continue
                
                target_text = entry.get('unmodified_original', '')
                model_output = str(entry.get('output', ''))
                
                if not target_text or not model_output:
                    continue
                    
                messages = build_judge_messages(target_text, model_output)
                llm_requests.append({
                    "exp_idx": exp_idx,
                    "entry_idx": entry_idx,
                    "messages": messages,
                    "target_text": target_text
                })

    if not all_experiments:
        print("No experiments found to evaluate.")
        return

    print(f"Total experiments loaded: {len(all_experiments)}")
    print(f"Total items requiring LLM judgement: {len(llm_requests)}")

    # 2. Run LLM on all requests in a single batch
    if not args.dry and llm_requests:
        from vllm import LLM, SamplingParams
        print(f"\nInitializing vLLM (model: {args.judge_model}, num_gpus: {args.num_gpus})")
        
        try:
            llm_kwargs = {
                "model": args.judge_model,
                "tensor_parallel_size": args.num_gpus,
                "trust_remote_code": True,
                "dtype": "bfloat16"
            }
            if args.max_model_length is not None:
                llm_kwargs["max_model_len"] = args.max_model_length
                
            llm = LLM(**llm_kwargs)
            tokenizer = llm.get_tokenizer()
        except Exception as e:
            print(f"Failed to load vLLM model: {e}")
            return
            
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        
        # Apply chat template and pre-tokenize using a fast ThreadPool to bypass the single-core bottleneck
        print(f"\nPre-processing and tokenizing {len(llm_requests)} prompts globally (Multithreaded)...")
        import concurrent.futures
        
        prompts_token_ids = [None] * len(llm_requests)
        
        def process_req(idx):
            req = llm_requests[idx]
            prompt_str = tokenizer.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
            return idx, tokenizer.encode(prompt_str)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(process_req, i) for i in range(len(llm_requests))]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Tokenizing Prompts"):
                idx, token_ids = future.result()
                prompts_token_ids[idx] = token_ids
            
        print(f"Submitting {len(prompts_token_ids)} tasks to vLLM execution engine...")
        outputs = llm.generate(prompt_token_ids=prompts_token_ids, sampling_params=sampling_params)
        
        # 3. Parse responses and populate summaries
        for idx, output in enumerate(outputs):
            response_text = output.outputs[0].text.strip().upper()
            req = llm_requests[idx]
            exp = all_experiments[req["exp_idx"]]
            entry = exp["entries"][req["entry_idx"]]
            
            # Very lenient check for "YES"
            is_recovered = "YES" in response_text
            
            if is_recovered:
                exp["summary"]["semantic_correct"] += 1
                exp["summary"]["recovered_cases"].append({
                    "id": entry.get('id'),
                    "score": 1.0, # Dummy highest score for visualization scripts
                    "target": req["target_text"],
                    "best_window": response_text # Store judge justification if any
                })
    elif args.dry:
        print("\n[DRY RUN] Skipping vLLM generation.")
        # Mock responses
        for req in llm_requests:
            exp = all_experiments[req["exp_idx"]]
            entry = exp["entries"][req["entry_idx"]]
            exp["summary"]["semantic_correct"] += 1
            exp["summary"]["recovered_cases"].append({
                    "id": entry.get('id'),
                    "score": 1.0,
                    "target": req["target_text"],
                    "best_window": "YES (DRY RUN)"
                })

    # 4. Save results to individual files
    print("\nSaving results...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for exp in all_experiments:
        summary = exp["summary"]
        total = summary["total_samples"]
        
        summary["original_accuracy"] = summary["original_correct"] / total if total > 0 else 0
        summary["semantic_accuracy"] = summary["semantic_correct"] / total if total > 0 else 0
        
        tech = exp["technique"]
        if args.dry:
            output_filename = f"{tech}_prompt_recovery_DRYRUN.json"
        else:
            output_filename = f"{tech}_prompt_recovery_{timestamp}.json"
            
        output_path = os.path.join(exp["output_dir"], output_filename)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"  Saved: {output_path} (Orig Acc: {summary['original_accuracy']:.2%}, Sem Acc: {summary['semantic_accuracy']:.2%})")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
