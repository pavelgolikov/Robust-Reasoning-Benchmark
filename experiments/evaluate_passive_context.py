import argparse
import glob
import json
import os
import random
import re
import time
from collections import defaultdict
from datasets import load_dataset
from util import extract_and_grade


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


def load_passive_source(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def pick_latest_file(files):
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def split_result_entries(data):
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError("Unsupported result JSON shape")

    entries = []
    summary = {}
    for item in data:
        if isinstance(item, dict) and "summary" in item and len(item) == 1:
            summary = item["summary"] or {}
        elif isinstance(item, dict) and "output" in item:
            entries.append(item)
    return entries, summary


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


def format_chat_text(tokenizer, system_prompt, user_prompt):
    if tokenizer is None:
        return f"{system_prompt}\n{user_prompt}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


_SOURCE_TOKEN_CACHE = {}


def source_token_ids(passive_source, tokenizer):
    """Tokenize the passive source once per (tokenizer, source); it is reused for every prompt."""
    key = (id(tokenizer), len(passive_source))
    if key not in _SOURCE_TOKEN_CACHE:
        if tokenizer is None:
            _SOURCE_TOKEN_CACHE[key] = passive_source.split()
        else:
            _SOURCE_TOKEN_CACHE[key] = tokenizer.encode(passive_source, add_special_tokens=False)
    return _SOURCE_TOKEN_CACHE[key]


def make_passive_body(passive_source, target_tokens, tokenizer, seed, mode="unique"):
    """Return (passive_body, repeat_factor).

    mode="unique": take one contiguous, non-repeating window of the source. Raises if the
    source is too short, so a corpus that would silently force repetition fails immediately.
    mode="repeat": rotate the source and repeat it until the target length is reached.
    """
    if target_tokens <= 0:
        return "", 0.0

    rng = random.Random(seed)
    source = " ".join(passive_source.split())
    words = source.split()
    if not words:
        raise ValueError("Passive source is empty after whitespace normalization.")

    if mode == "unique":
        token_ids = source_token_ids(source, tokenizer)
        if len(token_ids) < target_tokens:
            raise ValueError(
                f"Passive source has {len(token_ids)} tokens but {target_tokens} are needed for a "
                f"non-repeating body. Use a longer corpus or --passive_mode repeat."
            )
        offset = rng.randrange(len(token_ids) - target_tokens + 1)
        window = token_ids[offset:offset + target_tokens]
        body = " ".join(window) if tokenizer is None else decode_tokens(tokenizer, window)
        return body.strip(), 1.0

    if mode != "repeat":
        raise ValueError(f"Unknown passive mode: {mode}")

    offset = rng.randrange(len(words))
    rotated_words = words[offset:] + words[:offset]
    rotated = " ".join(rotated_words)
    repeated = ((rotated + "\n\n") * max(2, (target_tokens // max(1, len(words))) + 4)).strip()

    if tokenizer is None:
        repeated_words = repeated.split()
        body = " ".join(repeated_words[:target_tokens])
        factor = target_tokens / max(1, len(words))
        return body, factor

    token_ids = tokenizer.encode(repeated, add_special_tokens=False)
    if len(token_ids) < target_tokens:
        token_ids = (token_ids * ((target_tokens // max(1, len(token_ids))) + 2))[:target_tokens]
    else:
        token_ids = token_ids[:target_tokens]
    source_len = len(source_token_ids(source, tokenizer))
    return decode_tokens(tokenizer, token_ids).strip(), target_tokens / max(1, source_len)


def build_passive_prompt(target_problem, passive_body):
    """Mirror the compound prompt's shape: one instruction naming the whole task and the
    answer format, then labelled blocks. Compound opens with "Solve these completely
    unrelated math problems. For each problem put your final answer within \\boxed{}."
    and labels each block "Problem N:".
    """
    return f"""Read the historical passage below, then solve the math problem that follows it.
Put your final answer within \\boxed{{}}.

Historical passage:
{passive_body}

Problem:
{target_problem}
""".strip()


def find_compound_results_file(base_dir, safe_model_name, safe_dataset_name, num_distractors):
    results_dir = os.path.join(base_dir, "compound", "results", safe_model_name, safe_dataset_name)
    pattern = os.path.join(results_dir, "*.json")
    candidates = [
        path for path in glob.glob(pattern)
        if "_raw.json" not in os.path.basename(path)
        and os.path.basename(path).endswith(".json")
        and "compound" in os.path.basename(path)
    ]

    matching = []
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries, summary = split_result_entries(data)
            if not entries:
                continue
            found_distractors = summary.get("num_distractors")
            if found_distractors is None:
                prompt_problems = re.findall(r"Problem \d+:", entries[0].get("original", ""))
                found_distractors = max(0, len(prompt_problems) - 1)
            if int(found_distractors) == int(num_distractors):
                matching.append(path)
        except Exception:
            continue

    return pick_latest_file(matching)


def target_solution_start_char(output, target_problem_num):
    pattern = re.compile(r"Problem\s*(\d+)", re.IGNORECASE)
    matches = list(pattern.finditer(output))
    if not matches:
        raise ValueError("No 'Problem N' markers found in model output.")

    groups = []
    current_group = None
    for j, match in enumerate(matches):
        start = match.start()
        end = matches[j + 1].start() if j + 1 < len(matches) else len(output)
        length = end - start
        prob_num = int(match.group(1))

        if current_group is None:
            current_group = {"prob_num": prob_num, "start_idx": start, "total_len": length}
        elif current_group["prob_num"] == prob_num:
            current_group["total_len"] += length
        else:
            groups.append(current_group)
            current_group = {"prob_num": prob_num, "start_idx": start, "total_len": length}

    if current_group is not None:
        groups.append(current_group)

    target_groups = [group for group in groups if group["prob_num"] == target_problem_num]
    if not target_groups:
        raise ValueError(f"No Problem {target_problem_num} group found in model output.")

    return max(target_groups, key=lambda group: group["total_len"])["start_idx"]


def char_to_token_index(text, char_idx, tokenizer):
    if tokenizer is None:
        return len(text[:char_idx].split())

    try:
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        for token_idx, (start_char, end_char) in enumerate(encoding["offset_mapping"]):
            if end_char > char_idx:
                return token_idx
        return len(encoding["offset_mapping"])
    except Exception:
        return count_tokens(tokenizer, text[:char_idx])


def load_compound_pre_target_lengths(path, tokenizer):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries, summary = split_result_entries(data)

    by_id = defaultdict(list)
    all_lengths = []
    skipped = 0
    for entry in entries:
        original = entry.get("original", "")
        output = entry.get("output", "")
        system_prompt = entry.get("system_prompt", BASELINE_SYSTEM_PROMPT)
        prompt_problems = re.findall(r"Problem \d+:", original)
        if not prompt_problems:
            skipped += 1
            continue

        target_problem_num = len(prompt_problems)
        try:
            output_start_char = target_solution_start_char(output, target_problem_num)
        except ValueError:
            skipped += 1
            continue

        formatted_prompt = format_chat_text(tokenizer, system_prompt, original)
        full_text = formatted_prompt + output
        full_start_char = len(formatted_prompt) + output_start_char
        pre_target_tokens = char_to_token_index(full_text, full_start_char, tokenizer)
        problem_id = str(entry.get("id"))
        sample_idx = len(by_id[problem_id])
        info = {
            "sample_idx": sample_idx,
            "target_problem_num": target_problem_num,
            "pre_target_tokens": pre_target_tokens,
            "target_solution_start_char": full_start_char,
        }
        by_id[problem_id].append(info)
        all_lengths.append(pre_target_tokens)

    if not all_lengths:
        raise ValueError(f"No usable target-solution boundaries found in {path}")

    fallback = round(sum(all_lengths) / len(all_lengths))
    return by_id, fallback, skipped, summary


def build_context_matched_passive_prompt(
    target_problem,
    passive_source,
    tokenizer,
    desired_context_tokens,
    seed,
    passive_mode="unique",
):
    skeleton_prompt = build_passive_prompt(target_problem, "")
    skeleton_text = format_chat_text(tokenizer, BASELINE_SYSTEM_PROMPT, skeleton_prompt)
    skeleton_tokens = count_tokens(tokenizer, skeleton_text)
    passive_body_tokens_target = max(1, desired_context_tokens - skeleton_tokens)

    best = None
    for _ in range(4):
        passive_body, repeat_factor = make_passive_body(
            passive_source,
            passive_body_tokens_target,
            tokenizer,
            seed=seed,
            mode=passive_mode,
        )
        user_prompt = build_passive_prompt(target_problem, passive_body)
        formatted_prompt = format_chat_text(tokenizer, BASELINE_SYSTEM_PROMPT, user_prompt)
        actual_context_tokens = count_tokens(tokenizer, formatted_prompt)
        actual_passive_tokens = count_tokens(tokenizer, passive_body)
        diff = desired_context_tokens - actual_context_tokens
        best = (
            user_prompt,
            formatted_prompt,
            passive_body_tokens_target,
            actual_passive_tokens,
            actual_context_tokens,
            skeleton_tokens,
            repeat_factor,
        )
        if abs(diff) <= 2:
            break
        passive_body_tokens_target = max(1, passive_body_tokens_target + diff)

    return best


def main():
    parser = argparse.ArgumentParser(description="Equal-length passive-context control for intra-query degradation.")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--num_distractors", type=int, default=3, help="Match a compound run with this many pre-target math problems.")
    parser.add_argument(
        "--compound_results_file",
        type=str,
        default=None,
        help="Compound result JSON used to detect where target-problem solving starts. Defaults to the newest matching compound result.",
    )
    parser.add_argument(
        "--fallback_pre_target_tokens",
        type=int,
        default=None,
        help="Use this many pre-target context tokens if a target problem is missing from the compound boundary file.",
    )
    parser.add_argument("--passive_text_file", type=str, default=None)
    parser.add_argument(
        "--passive_mode",
        type=str,
        default="unique",
        choices=["unique", "repeat"],
        help="unique: one contiguous non-repeating window of the corpus (default). "
             "repeat: rotate and repeat a short excerpt, as in the original runs.",
    )
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
    default_passive_file = (
        "gibbon_decline_and_fall_long.txt"
        if args.passive_mode == "unique"
        else "gibbon_decline_and_fall_excerpt.txt"
    )
    passive_path = args.passive_text_file or os.path.join(
        base_dir,
        "passive_context",
        default_passive_file,
    )
    passive_source = load_passive_source(passive_path)

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

    compound_source_file = args.compound_results_file or find_compound_results_file(
        base_dir,
        safe_model_name,
        safe_dataset_name,
        args.num_distractors,
    )
    if compound_source_file is None:
        raise FileNotFoundError(
            "Could not find a matching compound result file. "
            "Provide --compound_results_file explicitly."
        )

    compound_lengths_by_id, fallback_pre_target_tokens, skipped_boundaries, compound_summary = load_compound_pre_target_lengths(
        compound_source_file,
        tokenizer,
    )
    if args.fallback_pre_target_tokens is not None:
        fallback_pre_target_tokens = args.fallback_pre_target_tokens

    print(f"Matching passive context to target-solution boundaries from: {compound_source_file}")
    print(
        f"Loaded boundaries for {len(compound_lengths_by_id)} problem ids; "
        f"fallback mean={fallback_pre_target_tokens}; skipped={skipped_boundaries}"
    )

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    source_dataset = load_dataset(args.dataset, split=args.split)
    dataset = source_dataset
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    prompts = []
    prompt_metadata = []

    for i, example in enumerate(dataset):
        problem_id = str(example.get("id", i))
        target_problem = sanitize_problem(example["problem"])
        boundary_options = compound_lengths_by_id.get(problem_id, [])

        for sample_idx in range(args.n_samples):
            if boundary_options:
                boundary = boundary_options[sample_idx % len(boundary_options)]
            else:
                boundary = {
                    "sample_idx": None,
                    "target_problem_num": args.num_distractors + 1,
                    "pre_target_tokens": fallback_pre_target_tokens,
                    "target_solution_start_char": None,
                }

            desired_context_tokens = int(boundary["pre_target_tokens"])
            (
                user_prompt,
                formatted_prompt_text,
                passive_body_tokens_target,
                actual_passive_tokens,
                actual_context_tokens,
                passive_prompt_skeleton_tokens,
                passive_repeat_factor,
            ) = build_context_matched_passive_prompt(
                target_problem,
                passive_source,
                tokenizer,
                desired_context_tokens,
                seed=args.seed + (i * max(1, args.n_samples)) + sample_idx,
                passive_mode=args.passive_mode,
            )

            if i == 0 and sample_idx == 0:
                print(f"\nSystem Prompt:\n{BASELINE_SYSTEM_PROMPT}")
                print(f"\nExample User Prompt:\n{user_prompt[:3000]}")
                print("-" * 30)
                print(f"Matched compound pre-target tokens: {desired_context_tokens}")
                print(f"Passive prompt skeleton tokens: {passive_prompt_skeleton_tokens}")
                print(f"Passive body target tokens: {passive_body_tokens_target}")
                print(f"Actual passive body tokens: {actual_passive_tokens}")
                print(f"Actual pre-generation context tokens: {actual_context_tokens}")
                print(f"Passive mode: {args.passive_mode}")
                print(f"Passive repeat factor: {passive_repeat_factor:.2f}")
                print(f"Passive source tokens available: {len(source_token_ids(' '.join(passive_source.split()), tokenizer))}")

            if not args.dry:
                formatted_prompt = formatted_prompt_text
            else:
                formatted_prompt = [
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]

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
                    "length_match_source": "compound_target_solution_boundary",
                    "compound_results_file": compound_source_file,
                    "compound_boundary_sample_idx": boundary["sample_idx"],
                    "compound_target_problem_num": boundary["target_problem_num"],
                    "compound_target_solution_start_char": boundary["target_solution_start_char"],
                    "matched_compound_pre_target_tokens": desired_context_tokens,
                    "passive_prompt_skeleton_tokens": passive_prompt_skeleton_tokens,
                    "passive_body_tokens_target": passive_body_tokens_target,
                    "actual_passive_body_tokens": actual_passive_tokens,
                    "actual_pre_generation_context_tokens": actual_context_tokens,
                    "passive_text_file": passive_path,
                    "passive_mode": args.passive_mode,
                    "passive_repeat_factor": passive_repeat_factor,
                    "fallback_pre_target_tokens": fallback_pre_target_tokens,
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
        "matched_compound_pre_target_tokens": 0,
        "actual_pre_generation_context_tokens": 0,
        "actual_passive_body_tokens": 0,
    }

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text if not args.dry else "placeholder output from dry run"
        meta = prompt_metadata[i]

        try:
            extracted, is_correct = extract_and_grade(generated_text, meta["ground_truth"], exp_name="compound")
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
        stats["matched_compound_pre_target_tokens"] += meta["matched_compound_pre_target_tokens"]
        stats["actual_pre_generation_context_tokens"] += meta["actual_pre_generation_context_tokens"]
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
                "length_match_source": meta["length_match_source"],
                "compound_results_file": meta["compound_results_file"],
                "compound_boundary_sample_idx": meta["compound_boundary_sample_idx"],
                "compound_target_problem_num": meta["compound_target_problem_num"],
                "compound_target_solution_start_char": meta["compound_target_solution_start_char"],
                "matched_compound_pre_target_tokens": meta["matched_compound_pre_target_tokens"],
                "passive_prompt_skeleton_tokens": meta["passive_prompt_skeleton_tokens"],
                "passive_body_tokens_target": meta["passive_body_tokens_target"],
                "actual_passive_body_tokens": meta["actual_passive_body_tokens"],
                "actual_pre_generation_context_tokens": meta["actual_pre_generation_context_tokens"],
                "passive_text_file": meta["passive_text_file"],
                "passive_mode": meta["passive_mode"],
                "passive_repeat_factor": meta["passive_repeat_factor"],
                "fallback_pre_target_tokens": meta["fallback_pre_target_tokens"],
                "output": generated_text,
                "extracted": extracted,
                "correct": is_correct,
                "output_tokens": output_tokens,
            }
        )

    acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
    avg_matched_pre_target = stats["matched_compound_pre_target_tokens"] / stats["total"] if stats["total"] else 0.0
    avg_actual_context = stats["actual_pre_generation_context_tokens"] / stats["total"] if stats["total"] else 0.0
    avg_passive = stats["actual_passive_body_tokens"] / stats["total"] if stats["total"] else 0.0

    print(f"\nAccuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")
    print(f"Failures: {stats['failures']}")
    print(f"Max Token Cutoffs: {stats['max_token_cutoffs']}")
    print(f"Avg matched compound pre-target tokens: {avg_matched_pre_target:.1f}")
    print(f"Avg actual pre-generation context tokens: {avg_actual_context:.1f}")
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
                "length_match_source": "compound_target_solution_boundary",
                "avg_matched_compound_pre_target_tokens": avg_matched_pre_target,
                "avg_actual_pre_generation_context_tokens": avg_actual_context,
                "avg_actual_passive_body_tokens": avg_passive,
                "passive_text_file": passive_path,
                "passive_mode": args.passive_mode,
                "avg_passive_repeat_factor": (
                    sum(m["passive_repeat_factor"] for m in prompt_metadata) / len(prompt_metadata)
                    if prompt_metadata
                    else 0.0
                ),
                "passive_source_tokens": len(
                    source_token_ids(" ".join(passive_source.split()), tokenizer)
                ),
                "compound_results_file": compound_source_file,
                "fallback_pre_target_tokens": fallback_pre_target_tokens,
                "compound_boundary_skipped": skipped_boundaries,
                "compound_summary": compound_summary,
                "task": "equal_length_passive_context",
            }
        }
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, "passive_context", "results", safe_model_name, safe_dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    run_id = (
        f"{safe_model_name}_{safe_dataset_name}_passive_context_{args.passive_mode}"
        f"_d{args.num_distractors}_s{args.seed}_{timestamp}"
    )
    json_file = os.path.join(output_dir, f"{run_id}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to: {json_file}")


if __name__ == "__main__":
    main()
