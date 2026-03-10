
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
    start_cache_renewal_thread,
)
from transformers import AutoTokenizer





def prepare_trimmed_context(context_type, args):
    """
    Load precomputed context from the specified file, and fix the system prompt.
    Returns (trimmed_messages, context_token_count, context_path).
    """
    context_path = args.context_file

    print(f"Loading context from {context_path} (Target: {args.context_size} tokens)...")
    if not os.path.exists(context_path):
        raise FileNotFoundError(f"Context file not found: {context_path}")

    with open(context_path, 'r') as f:
        trimmed = json.load(f)

    # Fix system prompt: always use BASELINE_SYSTEM_PROMPT as the system message
    if trimmed and trimmed[0]['role'] == 'system':
        if context_type == 'text':
            print(f"  Overriding system prompt (was: {trimmed[0]['content'][:60]}...)")
        trimmed[0]['content'] = BASELINE_SYSTEM_PROMPT
    else:
        print("  Inserting system prompt...")
        trimmed.insert(0, {'role': 'system', 'content': BASELINE_SYSTEM_PROMPT})

    token_count = args.context_size  # Precomputed target; verified exactly natively later
    print(f"  Loaded precomputed context: {len(trimmed)} messages, target: ~{token_count} tokens")
    return trimmed, token_count, context_path


def build_context_cache_from_trimmed(provider, trimmed_messages, model_name, context_type=None, context_size=None, ttl_seconds=7200):
    """
    Given already-trimmed context messages, create the appropriate provider cache.
    Returns context_cache dict {'type': ..., 'ref': ...}.
    """
    if provider == 'google':
        print("Checking for existing Google AI Studio context cache...")
        
        # Attempt to load from disk
        base = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(base, "context_caches")
        os.makedirs(cache_dir, exist_ok=True)
        
        safe_model = model_name.replace('/', '_')
        cache_record_file = os.path.join(cache_dir, f"google_cache_{safe_model}_{context_type}_{context_size}.json")
        
        if os.path.exists(cache_record_file):
            try:
                with open(cache_record_file, 'r') as f:
                    record = json.load(f)
                
                # Check expiration against our newly requested TTL
                # (If they ask for 48 hours but the existing cache only has 1 hr left, we should re-create)
                if time.time() < (record['created_at'] + record['ttl_seconds'] - ttl_seconds + 300):
                    print(f"Found active existing context cache: {record['cache_name']}")
                    return {'type': 'google', 'ref': record['cache_name']}
            except Exception as e:
                print(f"Error reading existing cache record: {e}")
                
        print("Creating new Google AI Studio context cache from trimmed context...")
        cache_name = create_google_context_cache_from_messages(trimmed_messages, model_name, ttl_seconds=ttl_seconds)
        
        # Save to disk
        try:
            import datetime
            creation_time = time.time()
            with open(cache_record_file, 'w') as f:
                json.dump({
                    "cache_name": cache_name,
                    "created_at": creation_time,
                    "created_at_human": datetime.datetime.fromtimestamp(creation_time).astimezone().isoformat(),
                    "expires_at_human": datetime.datetime.fromtimestamp(creation_time + ttl_seconds).astimezone().isoformat(),
                    "ttl_seconds": ttl_seconds,
                    "model_name": model_name,
                    "context_type": context_type,
                    "context_size": context_size
                }, f, indent=2)
        except Exception as e:
            print(f"Could not save cache record: {e}")
            
        return {'type': 'google', 'ref': cache_name}

    elif provider == 'anthropic':
        print("Preparing Anthropic prompt caching markers...")
        cached_msgs = prepare_anthropic_cached_messages_from_list(trimmed_messages)
        print(f"  {len(cached_msgs)} context messages marked for caching.")
        return {'type': 'anthropic', 'ref': cached_msgs}

    elif provider == 'openai':
        print(f"Preparing OpenAI context messages (implicit caching via exact prefix match)...")
        # For OpenAI, there is no explicit cache object, we just prepend the messages.
        return {'type': 'openai', 'ref': trimmed_messages}

    else:
        print(f"Warning: Context caching not implemented for provider '{provider}'. Context will not be sent.")
        return None


def run_context_eval_sequential(context_type, dataset, args, base_dir, timestamp):
    """Run sequential (non-batch) evaluation for a single context type."""
    print(f"\n{'='*60}")
    print(f"  Running context evaluation (sequential): {context_type.upper()}")
    print(f"{'='*60}")

    # Load, trim, and fix system prompt
    try:
        trimmed_context, context_token_count, context_path = prepare_trimmed_context(
            context_type, args
        )
    except Exception as e:
        print(f"Error preparing context: {e}")
        return None, None

    # Strip <think> and similar tags from all context messages in-place so the
    # preview, context_preview, and cache all see clean content.
    from api_utils import strip_thinking_tags
    for msg in trimmed_context:
        msg['content'] = strip_thinking_tags(msg['content'])

    common_context_str = "HF Local Tokenizer Used: None. Strict API Evaluation Enforced."
    
    # First 100 chars of the first distractor message (skip system at index 0)
    first_distractor = next((m['content'] for m in trimmed_context if m['role'] != 'system'), '')
    context_preview = first_distractor[:100]

    # STRICT TOKEN VALIDATION CHECK
    print(f"\n[PRE-FLIGHT] Verifying true context length against native API ({args.provider}:{args.model})...")
    import sys
    try:
        import asyncio
        from count_true_tokens import measure_tokens_native_check
        true_token_count = asyncio.run(measure_tokens_native_check(args.provider, args.model, trimmed_context))
        print(f"[PRE-FLIGHT] Native Token Count: {true_token_count:,} tokens (Target: {args.context_size:,})")
        
        # Require it to be strictly within 2% of the target context size
        margin = args.context_size * 0.02
        if abs(true_token_count - args.context_size) > margin:
            raise ValueError(f"CRITICAL: True '{args.model}' token count ({true_token_count:,}) deviates significantly from target ({args.context_size:,}). Aborting to save budget.")
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        print(f"[PRE-FLIGHT ERROR] Could not verify true token count natively: {e}. If this is a local test, continuing.")
        true_token_count = context_token_count

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

    print(f"\nPreparing to submit {len(jobs)} jobs for context_type='{context_type}' ({context_token_count} context tokens)...")
    cache_info = f"will be generated upon confirmation"

    if not getattr(args, 'no_preview', False):
        # Write a single randomly-picked full sample to a temp file for review before submitting
        import tempfile
        sample_job = random.choice(jobs)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, prefix='seq_preview_') as tmp:
            tmp_path = tmp.name
            tmp.write("=" * 80 + "\n")
            tmp.write("SEQUENTIAL SUBMISSION PREVIEW\n")
            tmp.write("=" * 80 + "\n\n")
            tmp.write(f"Model:           {args.model}\n")
            tmp.write(f"Provider:        {args.provider}\n")
            tmp.write(f"Context type:    {context_type}\n")
            tmp.write(f"Context source:  {context_path}\n")
            tmp.write(f"Context size:    {args.context_size} tokens (actual native count: ~{true_token_count:,} tokens)\n")
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

        user_input = input(f"\nSubmit {len(jobs)} jobs [{context_type}] sequentially to the cloud? Type 'Yes' to confirm: ")

        # Always delete the temp preview file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        if user_input.strip() != 'Yes':
            print("Skipping.")
            return None, None
    else:
        print(f"\nSkipping preview. Auto-submitting {len(jobs)} jobs [{context_type}] sequentially...")

    # Build context cache from the trimmed messages (always enabled)
    ttl = args.cache_ttl if args.cache_ttl > 0 else 7200
    context_cache = build_context_cache_from_trimmed(
        args.provider, trimmed_context, args.model,
        context_type=context_type, context_size=args.context_size,
        ttl_seconds=ttl
    )

    # Start auto-renewal thread for Google caches to prevent expiry during long runs
    cache_renewal_stop = None
    if context_cache and context_cache['type'] == 'google':
        _, cache_renewal_stop = start_cache_renewal_thread(
            context_cache['ref'], ttl_seconds=ttl
        )

    print(f"Generating answers for {len(jobs)} jobs...")
    results = []
    stats = {"correct": 0, "total": 0, "failures": 0}

    for i, job in enumerate(jobs):
        print(f"[{context_type}] Job {i+1}/{len(jobs)} (ID: {job['id']})...")
        try:
            generated_text = generate_response(
                job['messages'], args.model,
                provider=args.provider,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                context_cache=context_cache,
            )
        except Exception as e:
            # Reaches here only for safety filter blocks or exhausted transient retries
            # (quota/rate-limit errors are retried indefinitely inside generate_response)
            print(f"  Error (non-retryable): {e}")
            generated_text = f"ERROR: {str(e)}"

        extracted, is_correct = extract_and_grade(generated_text, job['ground_truth'])
        result_dict = {
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
        }
        results.append(result_dict)
        
        # Flush intermediate output securely behind loops for massive context iterations
        safe_model = args.model.replace('/', '_')
        safe_dataset = args.dataset.replace('/', '_')
        out_dir = os.path.join(base_dir, "context_saturation", "results", safe_model, safe_dataset)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"results_predef_{context_type}_{args.context_size}_{safe_model}_{timestamp}.json"
        intermediate_path = os.path.join(out_dir, filename + ".incomplete")
        
        with open(intermediate_path, 'a') as f:
            f.write(json.dumps(result_dict) + "\n")

        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1
        else:
            stats["failures"] += 1

        if args.sleep > 0 and i < len(jobs) - 1:
            print(f"  Pacing sequential request: Sleeping for {args.sleep}s to respect Cloud TPM limits...")
            import time
            time.sleep(args.sleep)

    # Stop cache renewal thread
    if cache_renewal_stop:
        cache_renewal_stop.set()
        print("  [Cache renewal] Stopped auto-renewal thread.")

    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"\n--- {context_type.upper()} Results ---")
    print(f"Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")

    # Save
    safe_model = args.model.replace('/', '_')
    safe_dataset = args.dataset.replace('/', '_')
    out_dir = os.path.join(base_dir, "context_saturation", "results", safe_model, safe_dataset)
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
    
    try:
        os.remove(intermediate_path)
    except OSError:
        pass
        
    return stats, out_path


def run_context_eval_batch(context_type, dataset, args, base_dir, timestamp):
    """Submit a batch job for a single context type."""
    print(f"\n{'='*60}")
    print(f"  Preparing batch: {context_type.upper()}")
    print(f"{'='*60}")

    # Load, trim, and fix system prompt
    try:
        trimmed_context, context_token_count, context_path = prepare_trimmed_context(
            context_type, args
        )
    except Exception as e:
        print(f"Error preparing context: {e}")
        return

    # Strip <think> and similar tags from all context messages in-place.
    from api_utils import strip_thinking_tags
    for msg in trimmed_context:
        msg['content'] = strip_thinking_tags(msg['content'])

    # First 100 chars of the first distractor message (skip system)
    first_distractor = next((m['content'] for m in trimmed_context if m['role'] != 'system'), '')
    context_preview = first_distractor[:100]

    # STRICT TOKEN VALIDATION CHECK
    print(f"\n[PRE-FLIGHT] Verifying true context length against native API ({args.provider}:{args.model})...")
    import sys
    try:
        import asyncio
        from count_true_tokens import measure_tokens_native_check
        true_token_count = asyncio.run(measure_tokens_native_check(args.provider, args.model, trimmed_context))
        print(f"[PRE-FLIGHT] Native Token Count: {true_token_count:,} tokens (Target: {args.context_size:,})")
        
        # Require it to be strictly within 2% of the target context size
        margin = args.context_size * 0.02
        if abs(true_token_count - args.context_size) > margin:
            raise ValueError(f"CRITICAL: True '{args.model}' token count ({true_token_count:,}) deviates significantly from target ({args.context_size:,}). Aborting to save budget.")
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        print(f"[PRE-FLIGHT ERROR] Could not verify true token count natively: {e}. If this is a local test, continuing.")
        true_token_count = context_token_count

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
    cache_info = f"will be generated upon confirmation"
    
    if not getattr(args, 'no_preview', False):
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
            tmp.write(f"Context source:  {context_path}\n")
            tmp.write(f"Context size:    {args.context_size} tokens (actual native count: ~{true_token_count:,} tokens)\n")
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
    else:
        print(f"\nSkipping preview. Auto-submitting {len(jobs)} jobs [{context_type}] to the cloud...")

    # Build context cache from the trimmed messages
    # Give batches a 48-hour cache TTL so they survive queue delays unless overridden
    ttl_seconds = args.cache_ttl if args.cache_ttl > 0 else 172800 
    context_cache = build_context_cache_from_trimmed(
        args.provider, trimmed_context, args.model,
        context_type=context_type, context_size=args.context_size,
        ttl_seconds=ttl_seconds
    )

    batch_info = submit_batch(
        jobs, args.model,
        provider=args.provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        context_cache=context_cache,
    )
    print(f"Batch submitted! Info: {batch_info}")

    safe_model = args.model.replace('/', '_')
    safe_dataset = args.dataset.replace('/', '_')
    exp_name = f"context_saturation_{context_type}"
    out_dir = os.path.join(base_dir, "context_saturation", "results", safe_model, safe_dataset)
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
        "temperature": args.temperature,
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
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature.")
    parser.add_argument("--provider", type=str, default=None,
                        help="API provider (google, anthropic, openai). Inferred from model name if omitted.")
    parser.add_argument("--batch", action="store_true",
                        help="Submit as async batch jobs instead of running sequentially.")
    parser.add_argument("--context_type", type=str, required=True,
                        help="Type of the context being evaluated (e.g. math or text) for consistent logging.")
    parser.add_argument("--context_file", type=str, required=True,
                        help="Exact path to the specific context JSON wrapper file to use.")
    parser.add_argument("--sleep", type=int, default=0,
                        help="Seconds to sleep between each sequential API job (e.g. 60) to avoid TPM rate limits.")
    parser.add_argument("--cache_ttl", type=int, default=0,
                        help="Exact cache TTL in seconds. Default 0 uses 7200s (seq) or 172800s (batch).")
    parser.add_argument("--no_preview", action="store_true",
                        help="Skip terminal preview and manual user validation prompt.")

    args = parser.parse_args()
    if not args.provider:
        args.provider = infer_provider(args.model)
    provider = args.provider

    random.seed(args.seed)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Model: {args.model} | Provider: {provider} | Batch: {args.batch} | context_size: {args.context_size}")
    print(f"Context file: {args.context_file} | Type: {args.context_type} | Caching: ENABLED")

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Dataset size: {len(dataset)} examples. Samples per problem: {args.n_samples}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_stats = {}
    all_paths = {}

    if args.batch:
        run_context_eval_batch(args.context_type, dataset, args, base_dir, timestamp)
    else:
        stats, out_path = run_context_eval_sequential(
            args.context_type, dataset, args, base_dir, timestamp
        )
        if stats is not None:
            all_stats[args.context_type] = stats
            all_paths[args.context_type] = out_path

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

        # # Save combined comparison
        # safe_model = args.model.replace('/', '_')
        # safe_dataset = args.dataset.replace('/', '_')
        # dirs = os.path.join(base_dir, "context_saturation", "results", safe_model, safe_dataset)
        # os.makedirs(dirs, exist_ok=True)
        # comparison_path = os.path.join(dirs, f"comparison_{args.context_size}_{safe_model}_{timestamp}.json")
        # with open(comparison_path, 'w') as f:
        #     json.dump({
        #         "metadata": {"model": args.model, "dataset": args.dataset,
        #                       "context_size": args.context_size, "context_types": list(all_stats.keys())},
        #         "comparison": {ct: {"statistics": st, "results_file": all_paths[ct]}
        #                        for ct, st in all_stats.items()},
        #     }, f, indent=2)
        # print(f"\nSaved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
