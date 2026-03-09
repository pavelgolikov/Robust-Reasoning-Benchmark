import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, remove_latex_comments, BASELINE_SYSTEM_PROMPT, extract_and_grade


# from trim_context import trim_context


def prepare_evaluation_data(tokenizer, trimmed_context, dataset, args):
    """
    Prepare all prompt token IDs and metadata upfront.
    Returns (all_inputs, metadata).
    """
    all_inputs = []
    metadata = []

    for i, example in enumerate(dataset):
        cleaned_problem = remove_latex_comments(example['problem'])
        user_prompt, _ = get_prompts(cleaned_problem, 'baseline')
        user_prompt = "Solve the following problem using regular mathematics.\n" + user_prompt

        full_conversation = trimmed_context + [{"role": "user", "content": user_prompt}]

        # Render then encode to IDs once
        prompt_str = tokenizer.apply_chat_template(full_conversation, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

        for sample_idx in range(args.n_samples):
            all_inputs.append(prompt_ids)
            metadata.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "post_context_prompt": user_prompt,
                "ground_truth": example['answer'],
            })
    return all_inputs, metadata


def run_context_eval(context_type, all_inputs, metadata, context_token_count, common_context_str, llm, sampling_params, args):
    """
    Run evaluation using pre-prepared token IDs.
    """
    print(f"\n{'='*60}")
    print(f"  Running context evaluation: {context_type.upper()}")
    print(f"{'='*60}")

    # Generate
    print(f"Generating answers for {len(all_inputs)} prompts using prompt_token_ids...")

    if not args.dry:
        # Pass IDs directly to vLLM to avoid redundant processing
        # Note: vLLM generate accepts prompt_token_ids as a list of lists of IDs.
        outputs = llm.generate(prompt_token_ids=all_inputs, sampling_params=sampling_params)
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
            "context_win_total": args.context_win_total,
            "distractor_token_count": context_token_count,
            "max_output_tokens": args.max_output_tokens,
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
    filename = f"results_predef_{context_type}_{context_token_count}_{safe_model}_{timestamp}.json"

    safe_dataset = args.dataset.replace('/', '_')
    dirs = f"context_saturation/results_context/{safe_model}/{safe_dataset}"
    os.makedirs(dirs, exist_ok=True)
    out_path = os.path.join(dirs, filename)

    final_output = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "context_math_file": args.context_math_file,
            "context_text_file": args.context_text_file,
            "context_token_count": context_token_count,
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
    parser.add_argument("--context_math_file", type=str, required=True, help="Path to pre-trimmed MATH context JSON")
    parser.add_argument("--context_text_file", type=str, required=True, help="Path to pre-trimmed TEXT context JSON")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--context_win_total", type=int, default=65536, help="Total context window capacity")
    parser.add_argument("--max_output_tokens", type=int, default=4096, help="Max generated tokens")
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
            max_model_len=args.context_win_total,
            dtype="bfloat16"
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_output_tokens)
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

    # 1. Load context files
    print(f"\n{'='*60}")
    print(f"  LOADING CONTEXTS")
    print(f"{'='*60}")
    
    prepped_contexts = {}
    context_paths = {'math': args.context_math_file, 'text': args.context_text_file}
    for context_type, path in context_paths.items():
        print(f"Loading {context_type} context from {path}...")
        with open(path, 'r') as f:
            prepped_contexts[context_type] = json.load(f)

    # 2. Prepare Evaluation Data for BOTH upfront (very important for performance)
    print(f"\n{'='*60}")
    print(f"  PREPARING EVALUATION DATA")
    print(f"{'='*60}")
    
    prepped_eval_data = {}
    for context_type in context_types:
        print(f"Preparing all prompts for '{context_type}' context...")
        trimmed = prepped_contexts[context_type]
        
        # Calculate context info for metadata
        common_context_str = tokenizer.apply_chat_template(trimmed, tokenize=False, add_generation_prompt=False)
        context_token_count = len(tokenizer.encode(common_context_str, add_special_tokens=False))
        
        inputs, meta = prepare_evaluation_data(tokenizer, trimmed, dataset, args)
        prepped_eval_data[context_type] = {
            "inputs": inputs,
            "metadata": meta,
            "token_count": context_token_count,
            "common_str": common_context_str
        }

    # Initialize all_stats and all_paths before the loop
    all_stats = {}
    all_paths = {}

    for context_type in context_types:
        data = prepped_eval_data[context_type]
        stats, results, out_path = run_context_eval(
            context_type=context_type,
            all_inputs=data["inputs"],
            metadata=data["metadata"],
            context_token_count=data["token_count"],
            common_context_str=data["common_str"],
            llm=llm,
            sampling_params=sampling_params,
            args=args
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
    comparison_path = os.path.join(dirs, f"comparison_{safe_model}_{timestamp}.json")

    comparison_output = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "context_math_file": args.context_math_file,
            "context_text_file": args.context_text_file,
            "context_types": list(all_stats.keys()),
        },
        "comparison": {ct: {"statistics": st, "results_file": all_paths[ct]} for ct, st in all_stats.items()},
    }

    with open(comparison_path, 'w') as f:
        json.dump(comparison_output, f, indent=2)
    print(f"\nSaved comparison to {comparison_path}")

if __name__ == "__main__":
    main()
