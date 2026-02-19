
import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, remove_latex_comments, last_boxed_only_string, remove_boxed, is_equiv
from api_utils import generate_response

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple experiments on AIME dataset (API Version)")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Name of the API model to evaluate")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of experiment names")
    parser.add_argument("--num_distractors", type=int, default=32, help="Number of distractors for split_indices")
    parser.add_argument("--provider", type=str, default=None, help="API Provider (google, openai, anthropic). Optional if model name implies it.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max output tokens. Defaults to 4096.")
    
    args = parser.parse_args()
    
    if args.names == 'all':
        experiment_names = [ 'context_saturation', 'interleaved_context_line', 'interleaved_context_word', 'interleaved_context_symbol',
        'not_not', 'opposites', 'sentence_reversal', 'word_reversal', 'wrappers', 'split_reversal' ]
    else:
        experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]

    print(f"Running experiments: {experiment_names}")

    # Load extracted variables if needed
    extracted_vars = {}
    if 'split_indices' in experiment_names:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            vars_path = os.path.join(base_dir, 'variables', 'extracted_terms_by_problem.json')
            with open(vars_path, 'r') as f:
                extracted_vars = json.load(f)
            for k, v in extracted_vars.items():
                extracted_vars[k] = [x.replace(" ", "_") for x in v]
            print("Variables loaded for split_indices.")
        except Exception as e:
            print(f"Warning: Failed to load extracted variables: {e}")

    random.seed(args.seed)

    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"Starting Multi-Evaluation on {len(dataset)} examples. Seed={args.seed}. Samples per problem={args.n_samples}")

    jobs = []
    
    for exp_name in experiment_names:
        print(f"Preparing prompts for: {exp_name}")
        
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
            
            # Create n samples
            for sample_idx in range(args.n_samples):
                jobs.append({
                    "experiment": exp_name,
                    "id": example.get('id', i),
                    "sample_idx": sample_idx,
                    "original": user_prompt,
                    "unmodified_original": example['problem'],
                    "system_prompt": system_prompt,
                    "messages": messages,
                    "ground_truth": ground_truth
                })

    # Generate
    print(f"Generating responses for {len(jobs)} jobs across {len(experiment_names)} experiments...")
    
    results_by_experiment = {name: [] for name in experiment_names}
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model.replace('/', '_').replace(' ', '_')
    safe_dataset_name = args.dataset.replace('/', '_')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Execute jobs
    total_jobs = len(jobs)
    for i, job in enumerate(jobs):
        try:
            print(f"Processing job {i+1}/{total_jobs} (Exp: {job['experiment']}, ID: {job['id']})...")
            # Call API
            generated_text = generate_response(
                job['messages'], 
                args.model, 
                provider=args.provider, 
                max_tokens=args.max_tokens
            )
        except Exception as e:
            print(f"Error generating for job {i}: {e}")
            generated_text = f"ERROR: {str(e)}"
            
        result_entry = {
            "id": job['id'],
            "system_prompt": job['system_prompt'],
            "original": job['original'],
            "unmodified_original": job['unmodified_original'],
            "ground_truth": job['ground_truth'],
            "output": generated_text,
            "extracted": None,
            "correct": None
        }
        results_by_experiment[job['experiment']].append(result_entry)

    # Save RAW results immediately (checkpointing)
    print("\nSaving RAW outputs to disk before parsing...")
    raw_files = {}
    for exp_name in experiment_names:
        experiment_dir = os.path.join(base_dir, exp_name)
        final_output_dir = os.path.join(experiment_dir, "results", safe_model_name, safe_dataset_name)
        os.makedirs(final_output_dir, exist_ok=True)
        run_id = f"{safe_model_name}_{safe_dataset_name}_{exp_name}_s{args.seed}_{timestamp}"
        
        raw_json_file = os.path.join(final_output_dir, f"{run_id}_raw.json")
        with open(raw_json_file, "w") as f:
            json.dump(results_by_experiment[exp_name], f, indent=2)
        print(f"  Saved raw outputs to: {raw_json_file}")
        raw_files[exp_name] = raw_json_file

    # Phase 2: Parse and Grade
    print(f"\nProcessing and Grading...")
    stats_by_experiment = {name: {"correct": 0, "total": 0, "failures": 0} for name in experiment_names}
    
    for exp_name in experiment_names:
        for entry in results_by_experiment[exp_name]:
            try:
                boxed_str = last_boxed_only_string(entry['output'])
                extracted = remove_boxed(boxed_str) if boxed_str else None
                try:
                    is_correct = is_equiv(extracted, entry['ground_truth'])
                except:
                    is_correct = False
            except Exception as e:
                print(f"Error processing sample {entry['id']}: {e}")
                extracted = f"ERROR: {str(e)}"
                is_correct = False
            
            entry['extracted'] = extracted
            entry['correct'] = is_correct
            
            stats_by_experiment[exp_name]["total"] += 1
            if is_correct:
                stats_by_experiment[exp_name]["correct"] += 1
            if extracted is None or (isinstance(extracted, str) and extracted.startswith("ERROR")):
                stats_by_experiment[exp_name]["failures"] += 1

    # Save Final Results
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
        
        run_id = f"{safe_model_name}_{safe_dataset_name}_{exp_name}_s{args.seed}_{timestamp}"
        experiment_dir = os.path.join(base_dir, exp_name)
        final_output_dir = os.path.join(experiment_dir, "results", safe_model_name, safe_dataset_name)
        os.makedirs(final_output_dir, exist_ok=True)
        
        json_file = os.path.join(final_output_dir, f"{run_id}.json")
        with open(json_file, "w") as f:
            json.dump(results_by_experiment[exp_name], f, indent=2)
        print(f"  Saved to: {json_file}")

        if exp_name in raw_files and os.path.exists(raw_files[exp_name]):
            try:
                os.remove(raw_files[exp_name])
                print(f"  Deleted raw checkpoint: {raw_files[exp_name]}")
            except OSError as e:
                print(f"  Warning: Could not delete raw checkpoint: {e}")

if __name__ == "__main__":
    main()
