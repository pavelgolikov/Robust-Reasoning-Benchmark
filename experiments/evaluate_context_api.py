
import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, remove_latex_comments, BASELINE_SYSTEM_PROMPT, extract_and_grade
from api_utils import generate_response
from trim_context import trim_context
from transformers import AutoTokenizer


def resolve_context_path(context_type):
    """Resolve the context file path for a given type ('math' or 'text')."""
    candidates = [
        f"experiments/context_{context_type}_1M.json",
        f"experiments/context_{context_type}.json",
        f"context_{context_type}_1M.json",
        f"context_{context_type}.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def run_context_eval(context_type, dataset, tokenizer, args):
    """
    Run evaluation for a single context type via API. Returns (stats_dict, results_list, out_path).
    """
    context_path = resolve_context_path(context_type)
    if context_path is None:
        print(f"Error: Context file not found for type '{context_type}'. Skipping.")
        return None, None, None

    print(f"\n{'='*60}")
    print(f"  Running context evaluation (API): {context_type.upper()}")
    print(f"{'='*60}")

    # Load and Truncate Context
    print(f"Trimming context to {args.context_size} tokens...")
    try:
        trimmed_context = trim_context(context_path, args.model, args.context_size, tokenizer=tokenizer)
    except Exception as e:
        print(f"Error trimming context: {e}")
        return None, None, None

    # Override System Prompt
    if trimmed_context and trimmed_context[0]['role'] == 'system':
        if context_type == 'text':
            print(f"Overriding system prompt (was: {trimmed_context[0]['content'][:80]}...)")
            trimmed_context[0]['content'] = BASELINE_SYSTEM_PROMPT
    else:
        print("Inserting system prompt...")
        trimmed_context.insert(0, {'role': 'system', 'content': BASELINE_SYSTEM_PROMPT})

    # Calculate context tokens once
    context_token_count = len(tokenizer.apply_chat_template(trimmed_context, tokenize=True, add_generation_prompt=False))
    common_context_str = tokenizer.apply_chat_template(trimmed_context, tokenize=False, add_generation_prompt=False)

    jobs = []

    print(f"Preparing {len(dataset)} examples for '{context_type}' context...")

    for i, example in enumerate(dataset):
        cleaned_problem = remove_latex_comments(example['problem'])

        user_prompt, _ = get_prompts(cleaned_problem, 'baseline')
        user_prompt = "Solve the following problem using regular mathematics.\n" + user_prompt

        full_conversation = list(trimmed_context) + [{"role": "user", "content": user_prompt}]

        for sample_idx in range(args.n_samples):
            jobs.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "post_context_prompt": user_prompt,
                "messages": full_conversation,
                "ground_truth": example['answer'],
            })

    # Generate
    print(f"Generating answers for {len(jobs)} jobs...")

    results = []
    stats = {"correct": 0, "total": 0, "failures": 0}

    total_jobs = len(jobs)
    for i, job in enumerate(jobs):
        print(f"[{context_type}] Processing job {i+1}/{total_jobs} (ID: {job['id']})...")
        try:
            generated_text = generate_response(job['messages'], args.model, max_tokens=args.max_tokens)
        except Exception as e:
            print(f"Error generating for job {i}: {e}")
            generated_text = f"ERROR: {str(e)}"

        output_len = len(generated_text)

        extracted, is_correct = extract_and_grade(generated_text, job['ground_truth'])

        results.append({
            "id": job['id'],
            "sample_idx": job.get('sample_idx', 0),
            "output": generated_text,
            "post_context_prompt": job['post_context_prompt'],
            "extracted": extracted,
            "ground_truth": job['ground_truth'],
            "correct": is_correct,
            "system_prompt": BASELINE_SYSTEM_PROMPT,
            "distractor_token_count": context_token_count,
            "model_output_len_char": output_len
        })

        stats["total"] += 1
        if is_correct: stats["correct"] += 1
        else: stats["failures"] += 1

    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"\n--- {context_type.upper()} Results ---")
    print(f"Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace('/', '_')
    filename = f"results_predef_{context_type}_{args.context_size}_{safe_model}_{timestamp}.json"

    safe_dataset = args.dataset.replace('/', '_')
    dirs = f"context_saturation/results_context/{safe_model}/{safe_dataset}"
    os.makedirs(dirs, exist_ok=True)
    out_path = os.path.join(dirs, filename)

    final_output = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "context_size": args.context_size,
            "context_type": context_type,
            "common_context": common_context_str
        },
        "statistics": stats,
        "results": results
    }

    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved {context_type} results to {out_path}")

    return stats, results, out_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate with Predefined Context Saturation - Math & Text (API Version)")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="API Model name")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="Dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--context_size", type=int, required=True, help="Target context size in tokens")
    parser.add_argument("--max_tokens", type=int, required=True, help="Max output tokens (required).")

    args = parser.parse_args()

    context_types = ['math', 'text']

    # Initialize Tokenizer (once, shared across both context types)
    print(f"Initializing tokenizer for context trimming (Target: {args.model})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except:
        print("Could not load specific tokenizer for model (expected for APIs). Falling back to 'gpt2' tokenizer for approximation.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load Dataset (once, shared across both context types)
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Run evaluation for each context type
    all_stats = {}
    all_paths = {}

    for context_type in context_types:
        stats, results, out_path = run_context_eval(
            context_type, dataset, tokenizer, args
        )
        if stats is not None:
            all_stats[context_type] = stats
            all_paths[context_type] = out_path

    # Print comparison summary
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Type':<8} | {'Accuracy':<18} | {'Correct':<10} | {'Total':<6}")
    print(f"{'-'*8}-+-{'-'*18}-+-{'-'*10}-+-{'-'*6}")

    for ct, st in all_stats.items():
        acc = st["correct"] / st["total"] if st["total"] > 0 else 0
        print(f"{ct:<8} | {acc:<18.2%} | {st['correct']:<10} | {st['total']:<6}")

    print()
    for ct, path in all_paths.items():
        print(f"  {ct}: {path}")

    # Save combined comparison
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace('/', '_')
    safe_dataset = args.dataset.replace('/', '_')
    dirs = f"context_saturation/results_context/{safe_model}/{safe_dataset}"
    os.makedirs(dirs, exist_ok=True)
    comparison_path = os.path.join(dirs, f"comparison_{args.context_size}_{safe_model}_{timestamp}.json")

    comparison_output = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "context_size": args.context_size,
            "context_types": list(all_stats.keys()),
        },
        "comparison": {ct: {"statistics": st, "results_file": all_paths[ct]} for ct, st in all_stats.items()},
    }

    with open(comparison_path, 'w') as f:
        json.dump(comparison_output, f, indent=2)
    print(f"\nSaved comparison to {comparison_path}")

if __name__ == "__main__":
    main()
