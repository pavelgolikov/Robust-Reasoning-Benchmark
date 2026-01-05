import os
import json
import re
import time
from opposites.transformation import apply_opposite_semantic_remapping
from opposites_not.transformation import apply_opposites_not_yot
from wrappers.transformation import apply_wrapper
from interleaved_context.transformation import apply_interleaved_context

import multiprocessing
import os
# Force 'spawn' to avoid CUDA re-initialization errors
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import random
import argparse
from datasets import load_dataset
from opposites.transformation import apply_opposite_semantic_remapping
import nltk

# Ensure NLTK data (WordNet) is available
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

def get_prompts(problem, name, extra_context=None):
    # modify problem according to experiment name
    if name == 'baseline':
        user_prompt = problem
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}. "
        return user_prompt, system_prompt
    elif name == 'opposites':
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}. \
            There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' blocks before user query."
        user_prompt = apply_opposite_semantic_remapping(problem, k=1)
        return user_prompt, system_prompt
    elif name == 'opposites_not':
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}. \
            There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' blocks before user query."
        user_prompt = apply_opposites_not_yot(problem, k_opp=1)
        return user_prompt, system_prompt
    elif name == 'wrappers':
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}. \
            There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' blocks before user query."
        user_prompt = apply_wrapper(problem, k=2)
        return user_prompt, system_prompt
    elif name == 'interleaved_context':
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}. \
            User query will consist of two problems - A and B, whose statements are interleaved. \
            You need to solve only problem A. If one problem statement is shorter than the other, \
            the empty lines resulting from the shorter problem statement will be filled with the \
            shorter problem statement repeated from the beginning."
        if extra_context is None:
            user_prompt = "Error: Missing extra context for interleaved transformation"
        else:
            user_prompt = apply_interleaved_context(problem, extra_context)
        return user_prompt, system_prompt
    else:
        return 'Not Implemented', ''

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
    parser.add_argument("--model", type=str, default="NONE", help="Path/Name of the model to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--name", type=str, default='baseline', help="Name of the experiment")
    parser.add_argument("--dry", action="store_true", help="Dry run - do not evaluate, only produce prompts")
    args = parser.parse_args()

    if not args.dry:
        print(f"Initializing vLLM with model: {args.model}")
        from vllm import LLM, SamplingParams
        max_model_length = 32000
        llm = LLM(
            model=args.model,
            tensor_parallel_size=4,
            trust_remote_code=True,
            max_model_len=max_model_length,
            dtype="bfloat16"
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=max_model_length)

    random.seed(args.seed)
    
    # Load Dataset
    print("Loading AIME 2024 dataset...")
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        
    print(f"Starting Evaluation on {len(dataset)} examples. Seed={args.seed}. Samples per problem={args.n_samples}")

    prompts = []
    metadata = []

    if not args.dry:
        tokenizer = llm.get_tokenizer()

    for i, example in enumerate(dataset):
        extra_context = None
        if args.name == 'interleaved_context':
            # Use next problem as context, wrapping around to the first for the last problem
            next_idx = (i + 1) % len(dataset)
            extra_context = dataset[next_idx]['problem']
            
        user_prompt, system_prompt = get_prompts(example['problem'], args.name, extra_context)
        ground_truth = example['answer']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if not args.dry:
            # Format prompt
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = messages
        
        # Create n independent samples for this problem
        for sample_idx in range(args.n_samples):
            prompts.append(formatted_prompt)
            metadata.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "original": user_prompt,
                "ground_truth": ground_truth
            })

    # Generate
    print(f"Generating responses for {len(prompts)} prompts...")
    if not args.dry:
        outputs = llm.generate(prompts, sampling_params)
    else:
        outputs = [''] * len(prompts)

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
        if not args.dry:
            generated_text = output.outputs[0].text
        else:
            generated_text = 'placeholder output from dry run'
        meta = metadata[i]
        
        extracted = extract_answer(generated_text)
        is_correct = normalize_answer(extracted) == normalize_answer(meta['ground_truth'])
        
        if is_correct:
            correct_count += 1
            
        result_entry = {
            "id": meta['id'],
            "system_prompt": system_prompt,
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
