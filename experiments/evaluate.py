import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, remove_latex_comments, extract_and_grade

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple experiments on AIME dataset (Efficiency Optimized)")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Path/Name of the model to evaluate")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of experiment names")
    parser.add_argument("--dry", action="store_true", help="Dry run - do not evaluate, only produce prompts")
    parser.add_argument("--num_distractors", type=int, default=32, help="Number of distractors for split_indices")
    parser.add_argument("--num_gpus", type=int, default=2, help="Num GPUs.")
    parser.add_argument("--max_model_length", type=int, default=32000, help="Max model length for vLLM")
    args = parser.parse_args()
    if args.names == 'all':
        experiment_names = [ 'context_saturation', 'interleaved_context_line', 'interleaved_context_word', 'interleaved_context_symbol',
        'not_not', 'opposites', 'sentence_reversal', 'word_reversal', 'wrappers', 'split_reversal' ]
    else:
        experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]

    # experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]
    print(f"Running experiments: {experiment_names}")

    # Load extracted variables if needed (if split_indices is in list)
    extracted_vars = {}
    if 'split_indices' in experiment_names:
        try:
            # Assume evaluate.py is in the same dir, so variables is in ./variables/
            base_dir = os.path.dirname(os.path.abspath(__file__))
            vars_path = os.path.join(base_dir, 'variables', 'extracted_terms_by_problem.json')
            with open(vars_path, 'r') as f:
                extracted_vars = json.load(f)
            # replace spaces with underscores
            for k, v in extracted_vars.items():
                extracted_vars[k] = [x.replace(" ", "_") for x in v]
            print("Variables loaded for split_indices.")
        except Exception as e:
            print(f"Warning: Failed to load extracted variables: {e}")

    # Initialize vLLM if not dry
    llm = None
    sampling_params = None
    tokenizer = None
    if not args.dry:
        print(f"Initializing vLLM with model: {args.model}")
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.num_gpus,
            trust_remote_code=True,
            max_model_len=args.max_model_length,
            dtype="bfloat16"
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_model_length)
        tokenizer = llm.get_tokenizer()

    random.seed(args.seed)

    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"Starting Multi-Evaluation on {len(dataset)} examples. Seed={args.seed}. Samples per problem={args.n_samples}")

    all_prompts = []
    # prompt_metadata will store info to map back to specific experiment/problem
    # Structure: list of dicts corresponding to all_prompts indices
    prompt_metadata = [] 

    for exp_name in experiment_names:
        print(f"Preparing prompts for: {exp_name}")
        # Identify next_idx context if needed
        # We process dataset again for each experiment
        
        for i, example in enumerate(dataset):
            extra_context = None
            if exp_name in ['interleaved_context_word', 'interleaved_context_line', 'interleaved_context_symbol']:
                next_idx = (i + 1) % len(dataset)
                extra_context = remove_latex_comments(dataset[next_idx]['problem'])
            
            prob_id = str(example.get('id', i))
            current_vars = extracted_vars.get(prob_id) if extracted_vars else None
            
            cleaned_problem = remove_latex_comments(example['problem'])

            user_prompt, system_prompt = get_prompts(
                cleaned_problem, 
                exp_name, 
                extra_context, 
                variables=current_vars,
                seed=args.seed, 
                num_distractors=args.num_distractors
            )
            ground_truth = example['answer']

            if i == 0:
                print("\n" + "-"*30)
                print(f"Experiment: {exp_name}")
                print(f"System Prompt:\n{system_prompt}")
                print(f"\nExample Problem Statement:\n{user_prompt}")
                print("-" * 30 + "\n")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            if not args.dry:
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = messages # Stored as list for dry run inspection
                
            # Create n samples
            for sample_idx in range(args.n_samples):
                all_prompts.append(formatted_prompt)
                prompt_metadata.append({
                    "experiment": exp_name,
                    "id": example.get('id', i),
                    "sample_idx": sample_idx,
                    "original": user_prompt,
                    "unmodified_original": example['problem'],
                    "system_prompt": system_prompt, # Capture system prompt too
                    "ground_truth": ground_truth
                })

    # Generate
    print(f"Generating responses for {len(all_prompts)} total prompts across {len(experiment_names)} experiments...")
    
    if not args.dry:
        # vLLM batch generation
        outputs = llm.generate(all_prompts, sampling_params)
    else:
        outputs = [''] * len(all_prompts)

    # Phase 1: Collect and Save Raw Outputs
    results_by_experiment = {name: [] for name in experiment_names}
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model.replace('/', '_').replace(' ', '_')
    safe_dataset_name = args.dataset.replace('/', '_')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for i, output in enumerate(outputs):
        if not args.dry:
            generated_text = output.outputs[0].text
        else:
            generated_text = "placeholder output from dry run"
            
        meta = prompt_metadata[i]
        exp = meta['experiment']
        
        result_entry = {
            "id": meta['id'],
            "system_prompt": meta['system_prompt'],
            "original": meta['original'],
            "unmodified_original": meta['unmodified_original'],
            "ground_truth": meta['ground_truth'],
            "output": generated_text,
            "extracted": None, # Placeholder
            "correct": None    # Placeholder
        }
        results_by_experiment[exp].append(result_entry)

    # Save RAW results immediately (checkpointing)
    print("\nSaving RAW outputs to disk before parsing...")
    raw_files = {}
    for exp_name in experiment_names:
        experiment_dir = os.path.join(base_dir, exp_name)
        # New hierarchy: results/<model>/<dataset>/
        final_output_dir = os.path.join(experiment_dir, "results", safe_model_name, safe_dataset_name)
        os.makedirs(final_output_dir, exist_ok=True)
        run_id = f"{safe_model_name}_{safe_dataset_name}_{exp_name}_s{args.seed}_{timestamp}"
        
        raw_json_file = os.path.join(final_output_dir, f"{run_id}_raw.json")
        with open(raw_json_file, "w") as f:
            json.dump(results_by_experiment[exp_name], f, indent=2)
        print(f"  Saved raw outputs to: {raw_json_file}")
        raw_files[exp_name] = raw_json_file

    # Phase 2: Parse and Grade
    print(f"\nProcessing and Grading {len(all_prompts)} responses...")
    stats_by_experiment = {name: {"correct": 0, "total": 0, "failures": 0} for name in experiment_names}
    
    for exp_name in experiment_names:
        for entry in results_by_experiment[exp_name]:
            try:
                extracted, is_correct = extract_and_grade(entry['output'], entry['ground_truth'])
            except Exception as e:
                print(f"Error processing sample {entry['id']}: {e}")
                extracted = f"ERROR: {str(e)}"
                is_correct = False
            
            # Update entry
            entry['extracted'] = extracted
            entry['correct'] = is_correct
            
            # Update stats
            stats_by_experiment[exp_name]["total"] += 1
            if is_correct:
                stats_by_experiment[exp_name]["correct"] += 1
            if extracted is None or (isinstance(extracted, str) and extracted.startswith("ERROR")):
                stats_by_experiment[exp_name]["failures"] += 1

    # Save Results
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n=== Multi-Eval Summary ===")
    for exp_name in experiment_names:
        stats = stats_by_experiment[exp_name]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"Experiment: {exp_name}")
        print(f"  Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")
        print(f"  Failures: {stats['failures']}")
        results_by_experiment[exp_name].append({
            "summary": {
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"],
                "failures": stats["failures"]
            }
        })
        
        # Save to file
        run_id = f"{safe_model_name}_{safe_dataset_name}_{exp_name}_s{args.seed}_{timestamp}"
        experiment_dir = os.path.join(base_dir, exp_name)
        # New hierarchy: results/<model>/<dataset>/
        final_output_dir = os.path.join(experiment_dir, "results", safe_model_name, safe_dataset_name)
        os.makedirs(final_output_dir, exist_ok=True)
        
        json_file = os.path.join(final_output_dir, f"{run_id}.json")
        with open(json_file, "w") as f:
            json.dump(results_by_experiment[exp_name], f, indent=2)
        print(f"  Saved to: {json_file}")

        # Reduce clutter: delete the raw file if the final file was successfully saved
        if exp_name in raw_files and os.path.exists(raw_files[exp_name]):
            try:
                os.remove(raw_files[exp_name])
                print(f"  Deleted raw checkpoint: {raw_files[exp_name]}")
            except OSError as e:
                print(f"  Warning: Could not delete raw checkpoint: {e}")

if __name__ == "__main__":
    main()
