import argparse
import difflib
import importlib
import json
import os
import random
import re
import time

from datasets import load_dataset


BASELINE_SYSTEM_PROMPT = (
    "You are a precise text reconstruction assistant. "
    "Do not solve math problems. Only recover the requested text.\n"
)

RECOVERED_OPEN = "<RECOVERED_PROBLEM>"
RECOVERED_CLOSE = "</RECOVERED_PROBLEM>"

TRANSFORMATION_NAMES = [
    # "not_not",
    # "opposites",
    # "wrappers",
    # "interleaved_context_line",
    # "interleaved_context_word",
    # "interleaved_context_symbol",
    # "sentence_reversal",
    # "word_reversal",
    # "split_reversal",
    # "rail_fence",
    "rectangle_perimeter",
    "snake_vertical",
    "snake_horizontal",
]

TECHNIQUE_DESCRIPTIONS = {
    "baseline": "No transformation applied. Recover the problem as it is presented in TRANSFORMED INPUT.",
    "not_not": "The input contains inserted double negations ('not not') before some numerical values or adjectives. Remove the redundant double negations to recover the original problem.",
    "opposites": "There are remapped terms in the input. The remappings are defined inside a defyn{} block. Apply the definitions to recover the original terms, and remove the defyn{} block.",
    "wrappers": "There are remapped wrapper expressions in the input. The remappings are defined inside a defyn{} block. Apply the definitions to unwrap the original terms, and remove the defyn{} block.",
    "interleaved_context_line": "The input consists of two problems, A and B, split into line segments of at most 60 symbols. Each segment is prefixed by a problem tag. Recover only the original statement of Problem A.",
    "interleaved_context_word": "The input consists of two problems, A and B, interleaved word by word. The first word belongs to Problem A, the second to Problem B, and so on. Recover only the original statement of Problem A.",
    "interleaved_context_symbol": "The input consists of two problems, A and B, interleaved symbol by symbol, including spaces and punctuation. The first symbol belongs to Problem A, the second to Problem B, and so on. Recover only the original statement of Problem A.",
    "sentence_reversal": "The order of sentences in the input has been reversed. Sentences are defined as sequences of symbols separated by periods. Reverse the sentence order to recover the original problem.",
    "word_reversal": "The order of words in the input has been reversed. Words are defined as sequences of symbols separated by spaces. Reverse the word order to recover the original problem.",
    "split_reversal": "Every word in the input has its symbols reversed. Words are defined as sequences of symbols separated by spaces. Reverse the symbols within each word to recover the original problem.",
    "rail_fence": "The input is encoded using the Rail Fence Cipher as a visual grid. Empty spaces are filled with dots. Read the characters in the zigzag rail-fence order to recover the original problem.",
    "rectangle_perimeter": "The input is mapped onto the perimeter of a rectangle. The message follows the perimeter clockwise beginning at the top-left. Recover the original problem from the grid.",
    "snake_vertical": "The input is written into a grid using a vertical snake pattern: down the first column, up the second column, and so on. Recover the original problem from the grid.",
    "snake_horizontal": "The input is written into a grid using a horizontal snake pattern: right across the first row, left across the second row, and so on. Recover the original problem from the grid.",
}


def remove_latex_comments(text):
    lines = text.split("\n")
    clean_lines = []

    def replacer(match):
        if match.group(1):
            return match.group(1)
        return ""

    for line in lines:
        clean_lines.append(re.sub(r"(\\\\|\\%)|(%.*)", replacer, line))
    return "\n".join(clean_lines)


def sanitize_inverted_escapes(text):
    return re.sub(r"([bntafr])\\", r"\1 \\", text)


def flatten_text(text):
    if text is None:
        return ""
    return text.replace("\n", "; ")


def sanitize_problem(text):
    return flatten_text(sanitize_inverted_escapes(remove_latex_comments(text)))


def normalize_for_recovery(text):
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)
    return text.strip()


def levenshtein_distance(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def char_error_rate(a, b, max_exact_chars=6000):
    if not a and not b:
        return 0.0, False
    denom = max(1, len(a))
    if len(a) <= max_exact_chars and len(b) <= max_exact_chars:
        return levenshtein_distance(a, b) / denom, False
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return 1.0 - ratio, True


def extract_recovered_text(output):
    if not output:
        return "", False
    pattern = re.compile(
        re.escape(RECOVERED_OPEN) + r"(.*?)" + re.escape(RECOVERED_CLOSE),
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(output)
    if matches:
        return matches[-1].strip(), True
    return output.strip(), False


def residual_status(recovered_norm, original_norm, transformed_norm, recovered, has_tags, threshold):
    if not has_tags:
        return "missing_recovered_tags"
    if recovered:
        return "recovered"
    if not recovered_norm:
        return "empty_recovered_text"

    transformed_cer, _ = char_error_rate(transformed_norm, recovered_norm)
    original_cer, _ = char_error_rate(original_norm, recovered_norm)

    if transformed_cer <= threshold:
        return "copied_transformed_input"
    if transformed_cer < original_cer:
        return "closer_to_transformed_than_original"
    if any(marker in recovered_norm for marker in ["defyn{", "GRID START", "<Problem A>", "<Problem B>"]):
        return "contains_transformation_markers"
    return "partial_or_other"


def apply_transformation(name, problem, extra_context=None, seed=None):
    if name == "baseline":
        return problem
    if name == "not_not":
        module = importlib.import_module("not_not.transformation")
        return module.apply_not_not(problem)
    if name == "opposites":
        module = importlib.import_module("opposites.transformation")
        return module.apply_opposites(problem, k=1)
    if name == "wrappers":
        module = importlib.import_module("wrappers.transformation")
        return module.apply_wrappers(problem, k=1)
    if name in {"interleaved_context_line", "interleaved_context_word", "interleaved_context_symbol"}:
        if extra_context is None:
            raise ValueError(f"{name} requires extra_context")
        module = importlib.import_module(f"{name}.transformation")
        return getattr(module, f"apply_{name}")(problem, extra_context)
    if name == "sentence_reversal":
        module = importlib.import_module("sentence_reversal.transformation")
        return module.apply_sentence_reversal(problem)
    if name == "word_reversal":
        module = importlib.import_module("word_reversal.transformation")
        return module.apply_word_reversal(problem)
    if name == "split_reversal":
        module = importlib.import_module("split_reversal.transformation")
        return module.apply_split_reversal(problem, separator=" ", func_name="reverse_string", seed=seed)
    if name == "rail_fence":
        module = importlib.import_module("rail_fence.transformation")
        return module.apply_rail_fence(problem, num_rails=3)
    if name == "rectangle_perimeter":
        module = importlib.import_module("rectangle_perimeter.transformation")
        return module.apply_rectangle_perimeter(problem)
    if name == "snake_vertical":
        module = importlib.import_module("snake_vertical.transformation")
        return module.apply_snake_vertical(problem)
    if name == "snake_horizontal":
        module = importlib.import_module("snake_horizontal.transformation")
        return module.apply_snake_horizontal(problem)
    raise ValueError(f"Unknown transformation: {name}")


def build_decode_prompt(name, transformed_problem):
    rule = TECHNIQUE_DESCRIPTIONS[name]
    return f"""YOUR TASK:
1. Read the TRANSFORMATION RULE and the TRANSFORMED INPUT.
2. Reverse the transformation and recover the original target problem statement.
3. Do not solve the math problem.
4. Output only the recovered problem statement between {RECOVERED_OPEN} and {RECOVERED_CLOSE}.

TRANSFORMATION RULE:
{rule}

TRANSFORMED INPUT:
{transformed_problem}
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Decode-only recovery experiment for RRB transformations.")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--names", type=str, default="all")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_model_length", type=int, default=32000)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--char_error_threshold", type=float, default=0.02)
    args = parser.parse_args()

    if args.names == "all":
        experiment_names = TRANSFORMATION_NAMES
    else:
        experiment_names = [name.strip() for name in args.names.split(",") if name.strip()]

    unknown = [name for name in experiment_names if name not in set(TRANSFORMATION_NAMES + ["baseline"])]
    if unknown:
        raise ValueError(f"Unknown transformation names: {unknown}")

    random.seed(args.seed)

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
            dtype="bfloat16",
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        tokenizer = llm.get_tokenizer()

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    source_dataset = load_dataset(args.dataset, split=args.split)
    dataset = source_dataset
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model.replace("/", "_").replace(" ", "_")
    safe_dataset_name = args.dataset.replace("/", "_")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for exp_name in experiment_names:
        print(f"\n{'=' * 60}")
        print(f"Decode-only recovery: {exp_name}")
        print(f"{'=' * 60}")

        prompts = []
        prompt_metadata = []

        for i, example in enumerate(dataset):
            canonical_original = sanitize_problem(example["problem"])
            extra_context = None
            if exp_name in {"interleaved_context_line", "interleaved_context_word", "interleaved_context_symbol"}:
                next_idx = (i + 1) % len(source_dataset)
                extra_context = sanitize_problem(source_dataset[next_idx]["problem"])

            transformed = apply_transformation(
                exp_name,
                canonical_original,
                extra_context=extra_context,
                seed=args.seed,
            )
            user_prompt = build_decode_prompt(exp_name, transformed)

            if i == 0:
                print(f"\nSystem Prompt:\n{BASELINE_SYSTEM_PROMPT}")
                print(f"\nExample User Prompt:\n{user_prompt[:3000]}")
                print("-" * 30)

            messages = [
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            if not args.dry:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted_prompt = messages

            for sample_idx in range(args.n_samples):
                prompts.append(formatted_prompt)
                prompt_metadata.append(
                    {
                        "id": example.get("id", i),
                        "sample_idx": sample_idx,
                        "original": user_prompt,
                        "transformed_input": transformed,
                        "unmodified_original": example["problem"],
                        "canonical_original": canonical_original,
                        "system_prompt": BASELINE_SYSTEM_PROMPT,
                        "ground_truth": example.get("answer"),
                    }
                )

        print(f"Generating responses for {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params) if not args.dry else [""] * len(prompts)

        results = []
        stats = {
            "total": 0,
            "recovered": 0,
            "exact_normalized_match": 0,
            "missing_tags": 0,
            "estimated_cer": 0,
            "max_token_cutoffs": 0,
        }
        residual_counts = {}

        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text if not args.dry else "placeholder output from dry run"
            meta = prompt_metadata[i]
            recovered_text, has_tags = extract_recovered_text(generated_text)

            original_norm = normalize_for_recovery(meta["canonical_original"])
            transformed_norm = normalize_for_recovery(meta["transformed_input"])
            recovered_norm = normalize_for_recovery(recovered_text)

            exact_match = recovered_norm == original_norm
            cer, estimated = char_error_rate(original_norm, recovered_norm)
            recovered = exact_match or cer <= args.char_error_threshold
            status = residual_status(
                recovered_norm,
                original_norm,
                transformed_norm,
                recovered,
                has_tags,
                args.char_error_threshold,
            )

            try:
                output_tokens = len(tokenizer.encode(generated_text)) if tokenizer else 0
            except Exception as exc:
                raise RuntimeError(f"Failed to count output tokens: {exc}") from exc

            stats["total"] += 1
            stats["recovered"] += int(recovered)
            stats["exact_normalized_match"] += int(exact_match)
            stats["missing_tags"] += int(not has_tags)
            stats["estimated_cer"] += int(estimated)
            if output_tokens >= args.max_tokens * 0.98:
                stats["max_token_cutoffs"] += 1
            residual_counts[status] = residual_counts.get(status, 0) + 1

            results.append(
                {
                    "id": meta["id"],
                    "sample_idx": meta["sample_idx"],
                    "system_prompt": meta["system_prompt"],
                    "original": meta["original"],
                    "transformed_input": meta["transformed_input"],
                    "unmodified_original": meta["unmodified_original"],
                    "canonical_original": meta["canonical_original"],
                    "ground_truth": meta["ground_truth"],
                    "output": generated_text,
                    "recovered_text": recovered_text,
                    "has_recovered_tags": has_tags,
                    "exact_normalized_match": exact_match,
                    "char_error_rate": cer,
                    "char_error_rate_estimated": estimated,
                    "recovered": recovered,
                    "residual_status": status,
                    "output_tokens": output_tokens,
                }
            )

        recovery_rate = stats["recovered"] / stats["total"] if stats["total"] else 0.0
        exact_rate = stats["exact_normalized_match"] / stats["total"] if stats["total"] else 0.0

        print(f"  Recovery rate: {recovery_rate:.2%} ({stats['recovered']}/{stats['total']})")
        print(f"  Exact normalized match: {exact_rate:.2%} ({stats['exact_normalized_match']}/{stats['total']})")
        print(f"  Missing tags: {stats['missing_tags']}")
        print(f"  Residual statuses: {residual_counts}")

        results.append(
            {
                "summary": {
                    "recovery_rate": recovery_rate,
                    "exact_normalized_match_rate": exact_rate,
                    "recovered": stats["recovered"],
                    "exact_normalized_match": stats["exact_normalized_match"],
                    "total": stats["total"],
                    "missing_tags": stats["missing_tags"],
                    "estimated_cer_count": stats["estimated_cer"],
                    "max_token_cutoffs": stats["max_token_cutoffs"],
                    "residual_status_counts": residual_counts,
                    "char_error_threshold": args.char_error_threshold,
                    "max_model_length": args.max_model_length,
                    "max_tokens": args.max_tokens,
                    "num_gpus": args.num_gpus,
                    "n_samples": args.n_samples,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "task": "decode_only_recovery",
                }
            }
        )

        final_output_dir = os.path.join(
            base_dir,
            "decode_recovery",
            "results",
            safe_model_name,
            safe_dataset_name,
        )
        os.makedirs(final_output_dir, exist_ok=True)
        run_id = f"{safe_model_name}_{safe_dataset_name}_{exp_name}_decode_recovery_s{args.seed}_{timestamp}"
        json_file = os.path.join(final_output_dir, f"{run_id}.json")
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to: {json_file}")


if __name__ == "__main__":
    main()
