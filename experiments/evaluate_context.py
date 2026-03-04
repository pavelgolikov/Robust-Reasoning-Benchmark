import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, remove_latex_comments, BASELINE_SYSTEM_PROMPT, extract_and_grade


from trim_context import trim_context


def resolve_context_path(context_type):
    """Resolve the context file path for a given type ('math' or 'text')."""
    # Try multiple naming conventions in both project-root and cwd-relative paths
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


def run_context_eval(context_type, dataset, tokenizer, llm, sampling_params, args):
    """
    Run evaluation for a single context type. Returns (stats_dict, results_list, out_path).
    """
    context_path = resolve_context_path(context_type)
    if context_path is None:
        print(f"Error: Context file not found for type '{context_type}'. Skipping.")
        return None, None, None

    print(f"\n{'='*60}")
    print(f"  Running context evaluation: {context_type.upper()}")
    print(f"{'='*60}")

    # Load and Truncate Context
    trimmed_context = trim_context(context_path, args.model, args.context_size, tokenizer=tokenizer)

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

    all_inputs = []
    metadata = []

    print(f"Preparing {len(dataset)} examples for '{context_type}' context...")

    for i, example in enumerate(dataset):
        cleaned_problem = remove_latex_comments(example['problem'])

        user_prompt, system_prompt = get_prompts(cleaned_problem, 'baseline')
        user_prompt = "Solve the following problem using regular mathematics.\n" + user_prompt

        full_conversation = trimmed_context + [{"role": "user", "content": user_prompt}]

        final_input_ids = tokenizer.apply_chat_template(full_conversation, tokenize=True, add_generation_prompt=True)

        for sample_idx in range(args.n_samples):
            all_inputs.append(final_input_ids)
            metadata.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "post_context_prompt": user_prompt,
                "ground_truth": example['answer'],
            })

    # Generate
    print(f"Generating answers for {len(all_inputs)} prompts...")

    if not args.dry:
        print("Decoding token sequences to text for inference...")
        decoded_prompts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in all_inputs]
        outputs = llm.generate(decoded_prompts, sampling_params=sampling_params)
    else:
        print("Dry run: Skipping generation.")
        outputs = []
        class MockOutput:
            def __init__(self, text):
                self.outputs = [type('obj', (object,), {'text': text, 'token_ids': [0]*10})]

        for _ in all_inputs:
            outputs.append(MockOutput("Mock Answer \\boxed{0}"))

    results = []
    stats = {"correct": 0, "total": 0, "failures": 0}

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        output_token_ids = output.outputs[0].token_ids
        output_len = len(output_token_ids)

        meta = metadata[i]

        extracted, is_correct = extract_and_grade(generated_text, meta['ground_truth'])

        results.append({
            "id": meta['id'],
            "sample_idx": meta.get('sample_idx', 0),
            "output": generated_text,
            "post_context_prompt": meta['post_context_prompt'],
            "extracted": extracted,
            "ground_truth": meta['ground_truth'],
            "correct": is_correct,
            "system_prompt": BASELINE_SYSTEM_PROMPT,
            "temperature": 0.7,
            "max_model_length": args.max_model_length,
            "distractor_token_count": context_token_count,
            "model_output_token_count": output_len
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
    parser = argparse.ArgumentParser(description="Evaluate with Predefined Context Saturation (Math & Text)")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Model name/path")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="Dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--context_size", type=int, required=True, help="Target context size in tokens")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_model_length", type=int, default=65536)
    parser.add_argument("--dry", action="store_true", help="Dry run")

    args = parser.parse_args()

    context_types = ['math', 'text']

    # Initialize vLLM (once, shared across both context types)
    llm = None
    sampling_params = None
    tokenizer = None

    if not args.dry:
        from vllm import LLM, SamplingParams
        print(f"Initializing vLLM with model: {args.model}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.num_gpus,
            trust_remote_code=True,
            max_model_len=args.max_model_length,
            dtype="bfloat16"
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_model_length)
        tokenizer = llm.get_tokenizer()
    else:
        print("Dry run: Skipping vLLM initialization. Loading tokenizer from huggingface...")
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except:
            print("Failed to load specific tokenizer, falling back to gpt2 for dry run token counting.")
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
            context_type, dataset, tokenizer, llm, sampling_params, args
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
