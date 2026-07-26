import argparse
import glob
import json
import os
import random
import re
import time
from collections import defaultdict

from datasets import load_dataset
from math_verify import parse, verify


BASELINE_SYSTEM_PROMPT = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n"


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


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = string.find("{", idx)
    if i < 0:
        return None

    num_left_braces_open = 1
    right_brace_idx = None
    i += 1
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s):
    match = re.match(r"\\(?:boxed|fbox)\s*\{(.*)\}$", s, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_and_grade(model_output, ground_truth):
    try:
        gold = parse(str(ground_truth))
    except Exception:
        return None, False

    boxed_str = last_boxed_only_string(model_output)
    boxed_val = remove_boxed(boxed_str) if boxed_str else None
    if boxed_val is not None and boxed_val.strip():
        try:
            return boxed_val, verify(gold, parse(boxed_val))
        except Exception:
            return boxed_val, False

    try:
        answer = parse(model_output)
        return str(answer) if answer else None, verify(gold, answer)
    except Exception:
        return None, False


def load_passive_source(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def pick_latest_file(files):
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_json_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported result JSON shape in {path}")


def find_prior_reasoning_file(base_dir, safe_model_name, safe_dataset_name):
    baseline_dir = os.path.join(base_dir, "baseline", "results", safe_model_name, safe_dataset_name)
    for subdir in ("compound", "perturb", ""):
        pattern = os.path.join(baseline_dir, subdir, "*.json") if subdir else os.path.join(baseline_dir, "*.json")
        candidates = [
            path for path in glob.glob(pattern)
            if not os.path.basename(path).endswith("_raw.json")
        ]
        latest = pick_latest_file(candidates)
        if latest:
            return latest
    return None


def load_prior_reasoning_lengths(path):
    per_id = defaultdict(list)
    all_lengths = []
    for entry in load_json_results(path):
        if not isinstance(entry, dict) or "summary" in entry:
            continue
        if entry.get("id") is None or entry.get("output_tokens") is None:
            continue
        try:
            output_tokens = int(entry["output_tokens"])
        except (TypeError, ValueError):
            continue
        problem_id = str(entry["id"])
        per_id[problem_id].append(output_tokens)
        all_lengths.append(output_tokens)

    if not all_lengths:
        raise ValueError(f"No usable output_tokens found in {path}")

    mean_by_id = {
        problem_id: round(sum(lengths) / len(lengths))
        for problem_id, lengths in per_id.items()
    }
    global_mean = round(sum(all_lengths) / len(all_lengths))
    return mean_by_id, global_mean


def resolve_prior_reasoning_source(args, base_dir, safe_model_name, safe_dataset_name):
    source_file = args.prior_reasoning_tokens_file
    if source_file is None:
        source_file = find_prior_reasoning_file(base_dir, safe_model_name, safe_dataset_name)

    if source_file:
        lengths_by_id, fallback_tokens = load_prior_reasoning_lengths(source_file)
        if args.fallback_prior_reasoning_tokens is not None:
            fallback_tokens = args.fallback_prior_reasoning_tokens
        return source_file, lengths_by_id, fallback_tokens

    if args.fallback_prior_reasoning_tokens is None:
        raise FileNotFoundError(
            "Could not find baseline output-token results for this model/dataset. "
            "Provide --prior_reasoning_tokens_file or --fallback_prior_reasoning_tokens."
        )
    return None, {}, args.fallback_prior_reasoning_tokens


def count_tokens(tokenizer, text):
    if tokenizer is None:
        return len(text.split())
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def decode_tokens(tokenizer, token_ids):
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def make_passive_body(passive_source, target_tokens, tokenizer, seed):
    if target_tokens <= 0:
        return ""

    rng = random.Random(seed)
    source = " ".join(passive_source.split())
    words = source.split()
    if not words:
        raise ValueError("Passive source is empty after whitespace normalization.")

    offset = rng.randrange(len(words))
    rotated_words = words[offset:] + words[:offset]
    rotated = " ".join(rotated_words)
    repeated = ((rotated + "\n\n") * max(2, (target_tokens // max(1, len(words))) + 4)).strip()

    if tokenizer is None:
        repeated_words = repeated.split()
        return " ".join(repeated_words[:target_tokens])

    token_ids = tokenizer.encode(repeated, add_special_tokens=False)
    if len(token_ids) < target_tokens:
        token_ids = (token_ids * ((target_tokens // max(1, len(token_ids))) + 2))[:target_tokens]
    else:
        token_ids = token_ids[:target_tokens]
    return decode_tokens(tokenizer, token_ids).strip()


def build_passive_prompt(target_problem, passive_body):
    return f"""Solve the following problem:

{target_problem}

Also take a look at this historical passage:

{passive_body}
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Equal-length passive-context control for intra-query degradation.")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--num_distractors", type=int, default=3, help="Match the generated reasoning length of this many pre-target math problems.")
    parser.add_argument(
        "--prior_reasoning_tokens_file",
        type=str,
        default=None,
        help="Baseline result JSON with per-problem output_tokens. Defaults to the newest baseline result for this model/dataset.",
    )
    parser.add_argument(
        "--fallback_prior_reasoning_tokens",
        type=int,
        default=None,
        help="Use this many tokens per missing prior problem when no matching baseline length is available.",
    )
    parser.add_argument(
        "--include_prior_problem_tokens",
        action="store_true",
        help="Also add the prompt-token length of the sampled prior problem statements to the passive context length.",
    )
    parser.add_argument("--passive_text_file", type=str, default=None)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_model_length", type=int, default=131072)
    parser.add_argument("--max_tokens", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    random.seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    safe_model_name = args.model.replace("/", "_").replace(" ", "_")
    safe_dataset_name = args.dataset.replace("/", "_")
    passive_path = args.passive_text_file or os.path.join(
        base_dir,
        "passive_context",
        "gibbon_decline_and_fall_excerpt.txt",
    )
    passive_source = load_passive_source(passive_path)
    prior_source_file, prior_lengths_by_id, fallback_prior_tokens = resolve_prior_reasoning_source(
        args,
        base_dir,
        safe_model_name,
        safe_dataset_name,
    )
    if prior_source_file:
        print(f"Matching passive context to prior generated reasoning lengths from: {prior_source_file}")
        print(f"Loaded token lengths for {len(prior_lengths_by_id)} problem ids; fallback mean={fallback_prior_tokens}")
    else:
        print(f"Matching passive context using fallback prior reasoning length: {fallback_prior_tokens} tokens/problem")

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

    prompts = []
    prompt_metadata = []

    for i, example in enumerate(dataset):
        target_problem = sanitize_problem(example["problem"])
        candidate_indices = [idx for idx in range(len(source_dataset)) if idx != i]
        sampled_indices = random.sample(
            candidate_indices,
            min(args.num_distractors, len(candidate_indices)),
        )
        sampled_distractors = [sanitize_problem(source_dataset[idx]["problem"]) for idx in sampled_indices]
        sampled_distractor_ids = [
            str(source_dataset[idx].get("id", idx))
            for idx in sampled_indices
        ]

        prior_reasoning_tokens_by_distractor = {
            problem_id: int(prior_lengths_by_id.get(problem_id, fallback_prior_tokens))
            for problem_id in sampled_distractor_ids
        }
        prior_reasoning_tokens_target = sum(prior_reasoning_tokens_by_distractor.values())

        math_context = "\n\n".join(
            f"Problem {j + 1}:\n{problem}"
            for j, problem in enumerate(sampled_distractors)
        )
        prior_problem_statement_tokens = count_tokens(tokenizer, math_context)
        passive_body_tokens_target = prior_reasoning_tokens_target
        if args.include_prior_problem_tokens:
            passive_body_tokens_target += prior_problem_statement_tokens
        passive_body_tokens_target = max(1, passive_body_tokens_target)
        passive_body = make_passive_body(
            passive_source,
            passive_body_tokens_target,
            tokenizer,
            seed=args.seed + i,
        )
        actual_passive_tokens = count_tokens(tokenizer, passive_body)
        user_prompt = build_passive_prompt(target_problem, passive_body)

        if i == 0:
            print(f"\nSystem Prompt:\n{BASELINE_SYSTEM_PROMPT}")
            print(f"\nExample User Prompt:\n{user_prompt[:3000]}")
            print("-" * 30)
            print(f"Matched prior reasoning tokens: {prior_reasoning_tokens_target}")
            print(f"Prior problem statement tokens: {prior_problem_statement_tokens}")
            print(f"Actual passive body tokens: {actual_passive_tokens}")

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
                    "unmodified_original": example["problem"],
                    "target_problem": target_problem,
                    "ground_truth": example["answer"],
                    "system_prompt": BASELINE_SYSTEM_PROMPT,
                    "sampled_distractor_ids_for_length": sampled_distractor_ids,
                    "prior_reasoning_tokens_by_distractor": prior_reasoning_tokens_by_distractor,
                    "matched_prior_reasoning_tokens": prior_reasoning_tokens_target,
                    "matched_prior_problem_statement_tokens": prior_problem_statement_tokens,
                    "passive_body_tokens_target": passive_body_tokens_target,
                    "actual_passive_body_tokens": actual_passive_tokens,
                    "passive_text_file": passive_path,
                    "prior_reasoning_tokens_file": prior_source_file,
                    "fallback_prior_reasoning_tokens": fallback_prior_tokens,
                    "include_prior_problem_tokens": args.include_prior_problem_tokens,
                }
            )

    print(f"Generating responses for {len(prompts)} passive-context prompts...")
    outputs = llm.generate(prompts, sampling_params) if not args.dry else [""] * len(prompts)

    results = []
    stats = {
        "correct": 0,
        "total": 0,
        "failures": 0,
        "max_token_cutoffs": 0,
        "matched_prior_reasoning_tokens": 0,
        "matched_prior_problem_statement_tokens": 0,
        "actual_passive_body_tokens": 0,
    }

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text if not args.dry else "placeholder output from dry run"
        meta = prompt_metadata[i]

        try:
            extracted, is_correct = extract_and_grade(generated_text, meta["ground_truth"])
        except Exception as exc:
            print(f"Error grading sample {meta['id']}: {exc}")
            extracted, is_correct = f"ERROR: {exc}", False

        try:
            output_tokens = count_tokens(tokenizer, generated_text)
        except Exception as exc:
            raise RuntimeError(f"Failed to count output tokens: {exc}") from exc

        stats["total"] += 1
        stats["correct"] += int(is_correct)
        stats["failures"] += int(extracted is None or (isinstance(extracted, str) and extracted.startswith("ERROR")))
        stats["max_token_cutoffs"] += int(output_tokens >= args.max_tokens * 0.98)
        stats["matched_prior_reasoning_tokens"] += meta["matched_prior_reasoning_tokens"]
        stats["matched_prior_problem_statement_tokens"] += meta["matched_prior_problem_statement_tokens"]
        stats["actual_passive_body_tokens"] += meta["actual_passive_body_tokens"]

        results.append(
            {
                "id": meta["id"],
                "sample_idx": meta["sample_idx"],
                "system_prompt": meta["system_prompt"],
                "original": meta["original"],
                "unmodified_original": meta["unmodified_original"],
                "target_problem": meta["target_problem"],
                "ground_truth": meta["ground_truth"],
                "sampled_distractor_ids_for_length": meta["sampled_distractor_ids_for_length"],
                "prior_reasoning_tokens_by_distractor": meta["prior_reasoning_tokens_by_distractor"],
                "matched_prior_reasoning_tokens": meta["matched_prior_reasoning_tokens"],
                "matched_prior_problem_statement_tokens": meta["matched_prior_problem_statement_tokens"],
                "passive_body_tokens_target": meta["passive_body_tokens_target"],
                "actual_passive_body_tokens": meta["actual_passive_body_tokens"],
                "passive_text_file": meta["passive_text_file"],
                "prior_reasoning_tokens_file": meta["prior_reasoning_tokens_file"],
                "fallback_prior_reasoning_tokens": meta["fallback_prior_reasoning_tokens"],
                "include_prior_problem_tokens": meta["include_prior_problem_tokens"],
                "output": generated_text,
                "extracted": extracted,
                "correct": is_correct,
                "output_tokens": output_tokens,
            }
        )

    acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
    avg_matched_reasoning = stats["matched_prior_reasoning_tokens"] / stats["total"] if stats["total"] else 0.0
    avg_matched_problem_statements = stats["matched_prior_problem_statement_tokens"] / stats["total"] if stats["total"] else 0.0
    avg_passive = stats["actual_passive_body_tokens"] / stats["total"] if stats["total"] else 0.0

    print(f"\nAccuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")
    print(f"Failures: {stats['failures']}")
    print(f"Max Token Cutoffs: {stats['max_token_cutoffs']}")
    print(f"Avg matched prior reasoning tokens: {avg_matched_reasoning:.1f}")
    print(f"Avg matched prior problem statement tokens: {avg_matched_problem_statements:.1f}")
    print(f"Avg actual passive body tokens: {avg_passive:.1f}")

    results.append(
        {
            "summary": {
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"],
                "failures": stats["failures"],
                "max_token_cutoffs": stats["max_token_cutoffs"],
                "max_model_length": args.max_model_length,
                "max_tokens": args.max_tokens,
                "num_distractors": args.num_distractors,
                "num_gpus": args.num_gpus,
                "n_samples": args.n_samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "avg_matched_prior_reasoning_tokens": avg_matched_reasoning,
                "avg_matched_prior_problem_statement_tokens": avg_matched_problem_statements,
                "avg_actual_passive_body_tokens": avg_passive,
                "passive_text_file": passive_path,
                "prior_reasoning_tokens_file": prior_source_file,
                "fallback_prior_reasoning_tokens": fallback_prior_tokens,
                "include_prior_problem_tokens": args.include_prior_problem_tokens,
                "task": "equal_length_passive_context",
            }
        }
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, "passive_context", "results", safe_model_name, safe_dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    run_id = f"{safe_model_name}_{safe_dataset_name}_passive_context_d{args.num_distractors}_s{args.seed}_{timestamp}"
    json_file = os.path.join(output_dir, f"{run_id}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {json_file}")


if __name__ == "__main__":
    main()
