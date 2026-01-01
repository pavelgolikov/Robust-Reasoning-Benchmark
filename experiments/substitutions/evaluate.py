import os
import json
import re
import time
import random
import argparse
from datasets import load_dataset

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM not found. This script requires vLLM for local inference.")
    LLM = None
    SamplingParams = None


def get_user_prompt(problem, name):
    # modify problem according to experiment name - for now only baseline is implemented
    if name == 'baseline':
        return problem
    else:
        return 'Unimplemented'

def extract_answer(text):
    if not text:
        return None
    
    # Priority 1: Boxed
    boxed_pattern = r"\\boxed\s*\{([^}]+)\}"
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()
        
    # Priority 2: Explicit answer statement
    answer_pattern = r"(?:The answer is|result is|so|equals)\s*[:=]?\s*(\d{1,4})(?:\.|,|\s|$)"
    matches = re.findall(answer_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1]
    
    return None

def normalize_answer(ans):
    if ans is None:
        return ""
    digits = "".join(filter(str.isdigit, str(ans)))
    if not digits:
        return ""
    return str(int(digits))

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on AIME dataset (Killarney/vLLM)")
    parser.add_argument("--model", type=str, default="GAIR/LIMO", help="Path/Name of the model to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--name", type=str, default='baseline', help="Name of the experiment")
    args = parser.parse_args()

    if LLM is None:
        raise ImportError("vLLM module is missing. Please install it.")

    print(f"Initializing vLLM with model: {args.model}")
    llm = LLM(model=args.model, tensor_parallel_size=1, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=12768)

    random.seed(args.seed)
    
    # Load Dataset
    print("Loading AIME 2024 dataset...")
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        
    print(f"Starting Evaluation on {len(dataset)} examples. Seed={args.seed}")

    # Prepare Prompts
    system_prompt = "You are a helpful math assistant. Solve the problem accurately. Output the final answer inside \\boxed{}."
    prompts = []
    metadata = []

    tokenizer = llm.get_tokenizer()

    for i, example in enumerate(dataset):
        user_prompt = get_user_prompt(example['problem'], args.name)
        ground_truth = example['answer']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Format prompt
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)
        
        metadata.append({
            "id": example.get('id', i),
            "original": user_prompt,
            "ground_truth": ground_truth
        })

    # Generate
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Process Results
    results = []
    correct_count = 0
    total = 0
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = args.model.replace('/', '_').replace(' ', '_')
    run_id = f"{safe_name}_{args.name}_s{args.seed}_{timestamp}"
    
    # Save to [experiment_name]/results logic
    experiment_dir = args.name
    final_output_dir = os.path.join(experiment_dir, "results")
    
    os.makedirs(final_output_dir, exist_ok=True)
    json_file = os.path.join(final_output_dir, f"{run_id}.json")

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        meta = metadata[i]
        
        extracted = extract_answer(generated_text)
        is_correct = normalize_answer(extracted) == normalize_answer(meta['ground_truth'])
        
        if is_correct:
            correct_count += 1
            
        result_entry = {
            "id": meta['id'],
            "original": meta['original'],
            "ground_truth": meta['ground_truth'],
            "output": generated_text,
            "extracted": extracted,
            "correct": is_correct
        }
        results.append(result_entry)
        total += 1
        
    accuracy = correct_count / total if total > 0 else 0
    print(f"\nEvaluation Complete. Accuracy: {accuracy:.2%} ({correct_count}/{total})")
    print(f"Results saved to: {json_file}")
    
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
