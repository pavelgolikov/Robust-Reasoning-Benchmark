
import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, remove_latex_comments, BASELINE_SYSTEM_PROMPT, extract_and_grade
from api_utils import (
    generate_response, submit_batch, infer_provider,
    create_google_context_cache_from_messages,
    prepare_anthropic_cached_messages_from_list,
)
from trim_context import trim_context
from transformers import AutoTokenizer


def resolve_context_path(context_type):
    """Resolve the context file path for a given type ('math' or 'text')."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, f"context_{context_type}_1M.json"),
        os.path.join(base, f"context_{context_type}.json"),
        f"context_{context_type}_1M.json",
        f"context_{context_type}.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def prepare_trimmed_context(context_type, args, tokenizer):
    """
    Load, trim to args.context_size tokens, and fix the system prompt.
    Returns (trimmed_messages, context_token_count, context_path).
    """
    context_path = resolve_context_path(context_type)
    if context_path is None:
        raise FileNotFoundError(f"Context file not found for type '{context_type}'.")

    print(f"Trimming '{context_type}' context to {args.context_size} tokens...")
    trimmed = trim_context(context_path, args.model, args.context_size, tokenizer=tokenizer)

    # Fix system prompt: always use BASELINE_SYSTEM_PROMPT as the system message
    if trimmed and trimmed[0]['role'] == 'system':
        if context_type == 'text':
            print(f"  Overriding system prompt (was: {trimmed[0]['content'][:60]}...)")
        trimmed[0]['content'] = BASELINE_SYSTEM_PROMPT
    else:
        print("  Inserting system prompt...")
        trimmed.insert(0, {'role': 'system', 'content': BASELINE_SYSTEM_PROMPT})

    token_count = len(tokenizer.apply_chat_template(trimmed, tokenize=True, add_generation_prompt=False))
    print(f"  Trimmed context: {len(trimmed)} messages, ~{token_count} tokens")
    return trimmed, token_count, context_path


def build_context_cache_from_trimmed(provider, trimmed_messages, model_name):
    """
    Given already-trimmed context messages, create the appropriate provider cache.
    Returns context_cache dict {'type': ..., 'ref': ...}.
    """
    if provider == 'google':
        print("Creating Google AI Studio context cache from trimmed context...")
        cache_name = create_google_context_cache_from_messages(trimmed_messages, model_name, ttl_seconds=7200)
        return {'type': 'google', 'ref': cache_name}

    elif provider == 'anthropic':
        print("Preparing Anthropic prompt caching markers...")
        cached_msgs = prepare_anthropic_cached_messages_from_list(trimmed_messages)
        print(f"  {len(cached_msgs)} context messages marked for caching.")
        return {'type': 'anthropic', 'ref': cached_msgs}

    else:
        print(f"Warning: Context caching not implemented for provider '{provider}'. Context will not be sent.")
        return None


def run_context_eval_sequential(context_type, dataset, args, tokenizer, base_dir, timestamp):
    """Run sequential (non-batch) evaluation for a single context type."""
    print(f"\n{'='*60}")
    print(f"  Running context evaluation (sequential): {context_type.upper()}")
    print(f"{'='*60}")

    # Load, trim, and fix system prompt
    try:
        trimmed_context, context_token_count, context_path = prepare_trimmed_context(
            context_type, args, tokenizer
        )
    except Exception as e:
        print(f"Error preparing context: {e}")
        return None, None

    common_context_str = tokenizer.apply_chat_template(trimmed_context, tokenize=False, add_generation_prompt=False)
    # First 100 chars of the first distractor message (skip system at index 0)
    first_distractor = next((m['content'] for m in trimmed_context if m['role'] != 'system'), '')
    context_preview = first_distractor[:100]

    # Build context cache from the trimmed messages (always enabled)
    context_cache = build_context_cache_from_trimmed(
        args.provider, trimmed_context, args.model
    )

    jobs = []
    for i, example in enumerate(dataset):
        cleaned_problem = remove_latex_comments(example['problem'])
        user_prompt, _ = get_prompts(cleaned_problem, 'baseline')
        user_prompt = "Solve the following problem using regular mathematics.\n" + user_prompt

        # Context is always delivered via cache; only send the new question
        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for sample_idx in range(args.n_samples):
            jobs.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "post_context_prompt": user_prompt,
                "messages": messages,
                "ground_truth": example['answer'],
                "system_prompt": BASELINE_SYSTEM_PROMPT,
            })

    print(f"Generating answers for {len(jobs)} jobs...")
    results = []
    stats = {"correct": 0, "total": 0, "failures": 0}

    for i, job in enumerate(jobs):
        print(f"[{context_type}] Job {i+1}/{len(jobs)} (ID: {job['id']})...")
        try:
            generated_text = generate_response(
                job['messages'], args.model,
                provider=args.provider,
                max_tokens=args.max_tokens,
                context_cache=context_cache,
            )
        except Exception as e:
            print(f"  Error: {e}")
            generated_text = f"ERROR: {str(e)}"

        extracted, is_correct = extract_and_grade(generated_text, job['ground_truth'])
        results.append({
            "id": job['id'],
            "sample_idx": job.get('sample_idx', 0),
            "output": generated_text,
            "post_context_prompt": job['post_context_prompt'],
            "extracted": extracted,
            "ground_truth": job['ground_truth'],
            "correct": is_correct,
            "context_type": context_type,
            "distractor_token_count": context_token_count,
            "model_output_len_char": len(generated_text),
            "system_prompt": BASELINE_SYSTEM_PROMPT,
            "context_preview": context_preview,
        })
        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1
        else:
            stats["failures"] += 1

    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"\n--- {context_type.upper()} Results ---")
    print(f"Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")

    # Save
    safe_model = args.model.replace('/', '_')
    safe_dataset = args.dataset.replace('/', '_')
    out_dir = os.path.join(base_dir, "context_saturation", "results_context", safe_model, safe_dataset)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"results_predef_{context_type}_{args.context_size}_{safe_model}_{timestamp}.json"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, 'w') as f:
        json.dump({
            "metadata": {
                "model": args.model,
                "dataset": args.dataset,
                "context_type": context_type,
                "context_size": args.context_size,
                "n_samples": args.n_samples,
                "context_cache": context_cache['type'] if context_cache else None,
                "common_context": common_context_str,
            },
            "statistics": stats,
            "results": results,
        }, f, indent=2)
    print(f"Saved {context_type} results to {out_path}")
    return stats, out_path


def run_context_eval_batch(context_type, dataset, args, tokenizer, base_dir, timestamp):
    """Submit a batch job for a single context type."""
    print(f"\n{'='*60}")
    print(f"  Preparing batch: {context_type.upper()}")
    print(f"{'='*60}")

    # Load, trim, and fix system prompt
    try:
        trimmed_context, context_token_count, context_path = prepare_trimmed_context(
            context_type, args, tokenizer
        )
    except Exception as e:
        print(f"Error preparing context: {e}")
        return

    # Build context cache from the trimmed messages
    context_cache = build_context_cache_from_trimmed(
        args.provider, trimmed_context, args.model
    )

    jobs = []
    for i, example in enumerate(dataset):
        cleaned_problem = remove_latex_comments(example['problem'])
        user_prompt, _ = get_prompts(cleaned_problem, 'baseline')
        user_prompt = "Solve the following problem using regular mathematics.\n" + user_prompt

        # Context is always delivered via cache; only send the new question
        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for sample_idx in range(args.n_samples):
            jobs.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "post_context_prompt": user_prompt,
                "messages": messages,
                "ground_truth": example['answer'],
                "unmodified_original": example['problem'],
                "context_type": context_type,
                "distractor_token_count": context_token_count,
                "system_prompt": BASELINE_SYSTEM_PROMPT,
                "context_preview": context_preview,
            })

    print(f"\nPreparing to submit {len(jobs)} jobs for context_type='{context_type}' ({context_token_count} context tokens)...")
    cache_info = f"{context_cache['type']} cache ({context_cache['ref'] if context_cache['type'] == 'google' else str(len(context_cache['ref'])) + ' msgs'})" if context_cache else "no cache"

    # Write a single randomly-picked full sample to a temp file for review before submitting
    import tempfile
    sample_job = random.choice(jobs)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, prefix='batch_preview_') as tmp:
        tmp_path = tmp.name
        tmp.write("=" * 80 + "\n")
        tmp.write("BATCH SUBMISSION PREVIEW\n")
        tmp.write("=" * 80 + "\n\n")
        tmp.write(f"Model:           {args.model}\n")
        tmp.write(f"Provider:        {args.provider}\n")
        tmp.write(f"Context type:    {context_type}\n")
        tmp.write(f"Context size:    {args.context_size} tokens (actual: ~{context_token_count} tokens)\n")
        tmp.write(f"Context msgs:    {len(trimmed_context)}\n")
        tmp.write(f"Cache:           {cache_info}\n")
        tmp.write(f"Total jobs:      {len(jobs)}\n")
        tmp.write(f"Max tokens:      {args.max_tokens}\n\n")
        tmp.write("=" * 80 + "\n")
        tmp.write("FULL SAMPLE (as it will be sent to the model):\n")
        tmp.write("=" * 80 + "\n\n")
        # Full context messages, untruncated
        for m in trimmed_context:
            tmp.write(f"[{m['role'].upper()}]\n{m['content']}\n\n")
        # The actual question appended after context
        tmp.write("-" * 80 + "\n")
        tmp.write("[USER]\n")
        tmp.write(sample_job['post_context_prompt'] + "\n\n")
        tmp.write(f"[GROUND TRUTH] {sample_job['ground_truth']}\n")
        tmp.write("=" * 80 + "\n")

    print(f"\nPreview written to: {tmp_path}")
    print("Open this file to review the context sample and example question.")

    user_input = input(f"\nSubmit {len(jobs)} jobs [{context_type}] to the cloud? Type 'Yes' to confirm: ")

    # Always delete the temp preview file
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    if user_input.strip() != 'Yes':
        print("Skipping.")
        return

    batch_info = submit_batch(
        jobs, args.model,
        provider=args.provider,
        max_tokens=args.max_tokens,
        context_cache=context_cache,
    )
    print(f"Batch submitted! Info: {batch_info}")

    safe_model = args.model.replace('/', '_')
    safe_dataset = args.dataset.replace('/', '_')
    exp_name = f"context_saturation_{context_type}"
    out_dir = os.path.join(base_dir, "context_saturation", "results_context", safe_model, safe_dataset)
    os.makedirs(out_dir, exist_ok=True)

    track_file = os.path.join(out_dir, f"batch_tracking_{timestamp}_{context_type}.json")
    jobs_file = os.path.join(out_dir, f"jobs_{batch_info['batch_id'].replace('/', '_')}_{context_type}.json")

    tracking_data = {
        "batch_id": batch_info["batch_id"],
        "provider": batch_info.get("provider", args.provider),
        "google_mode": batch_info.get("google_mode"),
        "model": args.model,
        "dataset": args.dataset,
        "experiment": exp_name,
        "context_type": context_type,
        "context_size": args.context_size,
        "context_token_count": context_token_count,
        "timestamp": timestamp,
        "max_tokens": args.max_tokens,
        "temperature": 0.7,
        "jobs_file": jobs_file,
        "status": batch_info.get("status", "SUBMITTED"),
        "metadata": batch_info,
    }

    with open(track_file, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    with open(jobs_file, 'w') as f:
        json.dump(jobs, f, indent=2)

    print(f"Saved tracking info to {track_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate with predefined context pollution (Math & Text) via API, with context caching."
    )
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="API model name")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--context_size", type=int, required=True,
                        help="Target context size in tokens (e.g. 250000 for 25%% of 1M context).")
    parser.add_argument("--max_tokens", type=int, required=True, help="Max output tokens.")
    parser.add_argument("--provider", type=str, default=None,
                        help="API provider (google, anthropic, openai). Inferred from model name if omitted.")
    parser.add_argument("--batch", action="store_true",
                        help="Submit as async batch jobs instead of running sequentially.")
    parser.add_argument("--context_types", type=str, default="math,text",
                        help="Comma-separated context types to run. Default: 'math,text'.")

    args = parser.parse_args()

    random.seed(args.seed)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    context_types = [t.strip() for t in args.context_types.split(',') if t.strip()]

    # Infer provider
    provider = args.provider or infer_provider(args.model)
    if not provider:
        raise ValueError(f"Cannot infer provider from model '{args.model}'. Specify --provider.")
    args.provider = provider

    print(f"Model: {args.model} | Provider: {provider} | Batch: {args.batch} | context_size: {args.context_size}")
    print(f"Context types: {context_types} | Caching: ENABLED")

    # Initialize tokenizer (shared for trimming)
    print("Initializing tokenizer for context trimming...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        print("  Could not load model-specific tokenizer. Falling back to gpt2 for token estimation.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Ensure the tokenizer has a chat template (needed by trim_context).
    # The gpt2 fallback has none, so inject a simple one for token counting.
    if not getattr(tokenizer, 'chat_template', None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] + ': ' + message['content'] + '\n' }}"
            "{% endfor %}"
        )

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Dataset size: {len(dataset)} examples. Samples per problem: {args.n_samples}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_stats = {}
    all_paths = {}

    for context_type in context_types:
        if args.batch:
            run_context_eval_batch(context_type, dataset, args, tokenizer, base_dir, timestamp)
        else:
            stats, out_path = run_context_eval_sequential(
                context_type, dataset, args, tokenizer, base_dir, timestamp
            )
            if stats is not None:
                all_stats[context_type] = stats
                all_paths[context_type] = out_path

    # Summary (sequential mode only)
    if not args.batch and all_stats:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY  (context_size={args.context_size})")
        print(f"{'='*60}")
        print(f"{'Type':<8} | {'Accuracy':<18} | {'Correct':<10} | {'Total':<6}")
        print(f"{'-'*8}-+-{'-'*18}-+-{'-'*10}-+-{'-'*6}")
        for ct, st in all_stats.items():
            acc = st["correct"] / st["total"] if st["total"] > 0 else 0
            print(f"{ct:<8} | {acc:<18.2%} | {st['correct']:<10} | {st['total']:<6}")
        for ct, path in all_paths.items():
            print(f"  {ct}: {path}")

        # Save combined comparison
        safe_model = args.model.replace('/', '_')
        safe_dataset = args.dataset.replace('/', '_')
        dirs = os.path.join(base_dir, "context_saturation", "results_context", safe_model, safe_dataset)
        os.makedirs(dirs, exist_ok=True)
        comparison_path = os.path.join(dirs, f"comparison_{args.context_size}_{safe_model}_{timestamp}.json")
        with open(comparison_path, 'w') as f:
            json.dump({
                "metadata": {"model": args.model, "dataset": args.dataset,
                              "context_size": args.context_size, "context_types": list(all_stats.keys())},
                "comparison": {ct: {"statistics": st, "results_file": all_paths[ct]}
                               for ct, st in all_stats.items()},
            }, f, indent=2)
        print(f"\nSaved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
