import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, remove_latex_comments, BASELINE_SYSTEM_PROMPT, extract_and_grade


def load_context(context_file):
    """Load a pre-built context JSON file (list of chat messages)."""
    if not os.path.exists(context_file):
        raise FileNotFoundError(f"Context file not found: {context_file}")

    with open(context_file, 'r') as f:
        messages = json.load(f)

    # Ensure system prompt is BASELINE_SYSTEM_PROMPT
    if messages and messages[0]['role'] == 'system':
        messages[0]['content'] = BASELINE_SYSTEM_PROMPT
    else:
        messages.insert(0, {'role': 'system', 'content': BASELINE_SYSTEM_PROMPT})

    print(f"  Loaded context: {len(messages)} messages from {context_file}")
    return messages


def prepare_inputs(dataset, trimmed_context, tokenizer, args):
    """
    For each dataset example, build the full conversation (context + question),
    render via chat template to a text string for vLLM.
    Returns (all_prompts, metadata).
    """
    all_prompts = []
    metadata = []

    for i, example in enumerate(dataset):
        cleaned_problem = remove_latex_comments(example['problem'])
        user_prompt, _ = get_prompts(cleaned_problem, 'baseline')
        user_prompt = "Solve the following problem using regular mathematics.\n" + user_prompt

        full_conversation = trimmed_context + [{"role": "user", "content": user_prompt}]

        # Render to text string via chat template (vLLM handles tokenization internally)
        rendered = tokenizer.apply_chat_template(
            full_conversation, tokenize=False, add_generation_prompt=True
        )

        if i == 0:
            input_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            print(f"\n--- Example Prompt (problem 0) ---")
            print(f"  Context messages: {len(trimmed_context)}")
            print(f"  User question: {user_prompt[:120]}...")
            print(f"  Total input tokens: {input_tokens}")
            print(f"  Ground truth: {example['answer']}")
            print(f"---\n")

        for sample_idx in range(args.n_samples):
            all_prompts.append(rendered)
            metadata.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "post_context_prompt": user_prompt,
                "ground_truth": example['answer'],
            })

    return all_prompts, metadata


def run_evaluation(all_prompts, metadata, context_token_count, llm, sampling_params, args):
    """Generate responses and grade them."""
    print(f"Generating answers for {len(all_prompts)} prompts...")

    if not args.dry:
        outputs = llm.generate(all_prompts, sampling_params=sampling_params)
    else:
        print("Dry run: Skipping generation.")
        outputs = []
        class MockOutput:
            def __init__(self, text):
                self.outputs = [type('obj', (object,), {'text': text, 'token_ids': [0] * 10})]
        for _ in all_prompts:
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
            "context_size": args.context_size,
            "distractor_token_count": context_token_count,
            "max_tokens": args.max_tokens,
            "model_output_token_count": output_len,
        })

        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1
        else:
            stats["failures"] += 1

    return results, stats


def save_results(results, stats, context_token_count, args, timestamp):
    """Save results JSON to disk."""
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"\n--- {args.context_type.upper()} Results ---")
    print(f"Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")

    safe_model = args.model.replace('/', '_')
    safe_dataset = args.dataset.replace('/', '_')
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "context_saturation", "results", safe_model, safe_dataset
    )
    os.makedirs(out_dir, exist_ok=True)

    filename = f"results_predef_{args.context_type}_{context_token_count}_{safe_model}_{timestamp}.json"
    out_path = os.path.join(out_dir, filename)

    final_output = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "context_file": args.context_file,
            "context_type": args.context_type,
            "context_size": args.context_size,
            "context_token_count": context_token_count,
            "n_samples": args.n_samples,
        },
        "statistics": stats,
        "results": results,
    }

    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved results to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate context saturation on local models via vLLM."
    )
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B",
                        help="HuggingFace model name/path")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024",
                        help="HuggingFace dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--context_file", type=str, required=True,
                        help="Path to pre-built context JSON file (list of chat messages).")
    parser.add_argument("--context_type", type=str, required=True,
                        help="Type of context (e.g. 'math' or 'text') for logging.")
    parser.add_argument("--context_size", type=int, required=True,
                        help="Target context size in tokens (used for validation and logging).")
    parser.add_argument("--max_tokens", type=int, default=32000,
                        help="Max output tokens for generation.")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None, help="Max model context length for vLLM. "
                             "Defaults to context_size + max_tokens + 512.")
    parser.add_argument("--dry", action="store_true", help="Dry run (skip model loading and generation)")
    args = parser.parse_args()

    if args.max_model_len is None:
        args.max_model_len = args.context_size + args.max_tokens + 512

    random.seed(args.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 1. Load tokenizer
    print(f"Loading tokenizer for model: {args.model}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    if tokenizer.chat_template is None:
        print(f"Warning: No chat template found for model '{args.model}'.")

    # 2. Load and validate context
    print(f"\n{'='*60}")
    print(f"  LOADING CONTEXT ({args.context_type.upper()})")
    print(f"{'='*60}")
    trimmed_context = load_context(args.context_file)

    # Measure actual token count
    context_rendered = tokenizer.apply_chat_template(
        trimmed_context, tokenize=False, add_generation_prompt=False
    )
    context_token_count = len(tokenizer.encode(context_rendered, add_special_tokens=False))
    print(f"  Context token count (native): {context_token_count:,} (target: {args.context_size:,})")

    margin = args.context_size * 0.10
    if abs(context_token_count - args.context_size) > margin:
        print(f"  WARNING: Context token count deviates >10% from target "
              f"({context_token_count:,} vs {args.context_size:,}). "
              f"The context file may have been built for a different tokenizer.")

    # 3. Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Dataset: {len(dataset)} examples. Samples per problem: {args.n_samples}")

    # 4. Prepare all inputs (tokenize upfront before loading model onto GPU)
    print(f"\nPreparing inputs...")
    all_prompts, metadata = prepare_inputs(dataset, trimmed_context, tokenizer, args)
    print(f"Prepared {len(all_prompts)} total prompts.")

    # 5. Initialize vLLM
    llm = None
    sampling_params = None

    if not args.dry:
        from vllm import LLM, SamplingParams
        print(f"\nInitializing vLLM: model={args.model}, gpus={args.num_gpus}, "
              f"max_model_len={args.max_model_len}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.num_gpus,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            dtype="bfloat16",
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    else:
        print("\nDry run: Skipping vLLM initialization.")

    # 6. Run evaluation
    print(f"\n{'='*60}")
    print(f"  RUNNING EVALUATION: {args.context_type.upper()}")
    print(f"{'='*60}")
    results, stats = run_evaluation(
        all_prompts, metadata, context_token_count, llm, sampling_params, args
    )

    # 7. Save results
    out_path = save_results(results, stats, context_token_count, args, timestamp)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"  Context type: {args.context_type}")
    print(f"  Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")
    print(f"  Results: {out_path}")


if __name__ == "__main__":
    main()