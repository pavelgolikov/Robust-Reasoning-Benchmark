
import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, remove_latex_comments, extract_and_grade
from api_utils import generate_response, submit_batch

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple experiments on AIME dataset (API Version)")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Name of the API model to evaluate")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of experiment names")
    parser.add_argument("--provider", type=str, default=None, help="API Provider (google, openai, anthropic). Optional if model name implies it.")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max output tokens (required to avoid accidental truncation).")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature.")
    parser.add_argument("--batch", action="store_true", help="Submit as an async batch job instead of running sequentially")
    
    args = parser.parse_args()
    
    if args.names == 'all':
        experiment_names = [ 'baseline',
            'interleaved_context_line',
            'interleaved_context_word',
            'interleaved_context_symbol',
            'not_not',
            'opposites',
            'sentence_reversal',
            'word_reversal',
            'wrappers',
            'split_reversal',
            'rail_fence',
            'compound' ]
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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model.replace('/', '_').replace(' ', '_')
    safe_dataset_name = args.dataset.replace('/', '_')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Process each transformation independently: prepare -> generate -> grade -> save
    for exp_name in experiment_names:
        print(f"\n{'='*60}")
        print(f"Processing transformation: {exp_name}")
        print(f"{'='*60}")

        # 1. Prepare jobs for this transformation
        jobs = []
        
        for i, example in enumerate(dataset):
            extra_context = None
            if exp_name in ['interleaved_context_word', 'interleaved_context_line', 'interleaved_context_symbol']:
                next_idx = (i + 1) % len(dataset)
                extra_context = remove_latex_comments(dataset[next_idx]['problem'])
            elif exp_name == 'compound':
                pre_target_count = 3
                indices = random.sample([idx for idx in range(len(dataset)) if idx != i], min(pre_target_count, max(1, len(dataset)-1)))
                extra_context = [remove_latex_comments(dataset[idx]['problem']) for idx in indices]
            
            prob_id = str(example.get('id', i))
            current_vars = extracted_vars.get(prob_id) if extracted_vars else None
            
            cleaned_problem = remove_latex_comments(example['problem'])

            user_prompt, system_prompt = get_prompts(
                cleaned_problem, 
                exp_name, 
                extra_context, 
                variables=current_vars,
                seed=args.seed, 
            )
            ground_truth = example['answer']

            # if i == 0:
            #     print(f"\nSystem Prompt:\n{system_prompt}")
            #     print(f"\nExample Problem Statement:\n{user_prompt}")
            #     print("-" * 30)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            for sample_idx in range(args.n_samples):
                jobs.append({
                    "id": example.get('id', i),
                    "sample_idx": sample_idx,
                    "original": user_prompt,
                    "unmodified_original": example['problem'],
                    "system_prompt": system_prompt,
                    "messages": messages,
                    "ground_truth": ground_truth
                })

        # 2. Generate responses for this transformation
        if args.batch:
            print(f"\nPreparing to submit {len(jobs)} jobs as a batch...")
            samples = random.sample(jobs, min(2, len(jobs)))
            print("\n" + "="*40 + " BATCH SUBMISSION PREVIEW " + "="*40)
            print(f"Model: {args.model}")
            print(f"Transformation: {exp_name}")
            print(f"Total Jobs: {len(jobs)}")
            print("\n--- Example 1 ---")
            print(f"System Prompt:\n{samples[0]['system_prompt']}\n")
            print(f"User Prompt:\n{samples[0]['original']}\n")
            if len(samples) > 1:
                print("\n--- Example 2 ---")
                print(f"System Prompt:\n{samples[1]['system_prompt']}\n")
                print(f"User Prompt:\n{samples[1]['original']}\n")
            print("="*106 + "\n")
            
            user_input = input(f"Do you want to submit this batch of {len(jobs)} jobs to the cloud? Type 'Yes' to confirm: ")
            if user_input.strip() != "Yes":
                print("Skipping batch submission.")
                continue
                
            batch_info = submit_batch(jobs, args.model, provider=args.provider, max_tokens=args.max_tokens, temperature=args.temperature)
            print(f"Batch submitted successfully! Info: {batch_info}")
            print(f"Max Tokens: {args.max_tokens}, Temperature: {args.temperature}")
            
            experiment_dir = os.path.join(base_dir, exp_name)
            final_output_dir = os.path.join(experiment_dir, "results", safe_model_name, safe_dataset_name)
            os.makedirs(final_output_dir, exist_ok=True)
            
            track_file = os.path.join(final_output_dir, f"batch_tracking_{timestamp}.json")
            jobs_file = os.path.join(final_output_dir, f"jobs_{batch_info['batch_id'].replace('/', '_')}.json")
            
            tracking_data = {
                "batch_id": batch_info["batch_id"],
                "provider": batch_info.get("provider", args.provider),
                "model": args.model,
                "dataset": args.dataset,
                "experiment": exp_name,
                "timestamp": timestamp,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "jobs_file": jobs_file,
                "status": batch_info.get("status", "SUBMITTED"),
                "metadata": batch_info
            }
            
            with open(track_file, "w") as f:
                json.dump(tracking_data, f, indent=2)
                
            with open(jobs_file, "w") as f:
                json.dump(jobs, f, indent=2)
                
            print(f"Saved batch tracking info to {track_file}")
            continue

        print(f"Generating responses for {len(jobs)} jobs...")
        
        results = []
        for i, job in enumerate(jobs):
            try:
                print(f"  Job {i+1}/{len(jobs)} (ID: {job['id']})...")
                generated_text = generate_response(
                    job['messages'], 
                    args.model, 
                    provider=args.provider, 
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                )
            except Exception as e:
                print(f"  Error generating for job {i}: {e}")
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
            results.append(result_entry)

        # 3. Save RAW results (checkpoint before grading)
        experiment_dir = os.path.join(base_dir, exp_name)
        final_output_dir = os.path.join(experiment_dir, "results", safe_model_name, safe_dataset_name)
        os.makedirs(final_output_dir, exist_ok=True)
        run_id = f"{safe_model_name}_{safe_dataset_name}_{exp_name}_s{args.seed}_{timestamp}"
        
        raw_json_file = os.path.join(final_output_dir, f"{run_id}_raw.json")
        with open(raw_json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved raw checkpoint: {raw_json_file}")

        # 4. Grade results
        stats = {"correct": 0, "total": 0, "failures": 0}
        for entry in results:
            try:
                extracted, is_correct = extract_and_grade(entry['output'], entry['ground_truth'])
            except Exception as e:
                print(f"Error processing sample {entry['id']}: {e}")
                extracted = f"ERROR: {str(e)}"
                is_correct = False
            
            entry['extracted'] = extracted
            entry['correct'] = is_correct
            
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1
            if extracted is None or (isinstance(extracted, str) and extracted.startswith("ERROR")):
                stats["failures"] += 1

        # 5. Save final graded results
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"\n  Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")
        print(f"  Failures: {stats['failures']}")
        print(f"  Max Tokens: {args.max_tokens}, Temperature: {args.temperature}")
        
        results.append({
            "summary": {
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"],
                "failures": stats["failures"],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature
            }
        })
        
        json_file = os.path.join(final_output_dir, f"{run_id}.json")
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to: {json_file}")

        # Cleanup raw checkpoint
        if os.path.exists(raw_json_file):
            try:
                os.remove(raw_json_file)
            except OSError as e:
                print(f"  Warning: Could not delete raw checkpoint: {e}")

if __name__ == "__main__":
    main()
