
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

# Constants
MODEL_PATH = "rstar2-reproduce/rStar2-Agent-14B"

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

def run_local_evaluation(
    experiment_name,
    transformation_function,
    system_prompt,
    results_dir,
    logs_dir,
    limit=None,
    k=None,
    seed=42
):
    """
    Run evaluation specifically for Killarney/Local HPC using vLLM.
    """
    if LLM is None:
        raise ImportError("vLLM module is missing. Please install it.")

    print(f"Initializing vLLM with model: {MODEL_PATH}")
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=8192)

    if seed is not None:
        random.seed(seed)
        
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        
    print(f"Starting {experiment_name} on {len(dataset)} examples. Seed={seed}, k={k}")
    
    # Pre-calculate transformations to batch generate
    prompts = []
    metadata = []
    
    for i, example in enumerate(dataset):
        problem = example['problem']
        ground_truth = example['answer']
        
        # Apply transformation
        if transformation_function:
            try:
                problem_input, remapping = transformation_function(problem, k=k, seed=seed)
            except TypeError:
                # Handle legacy functions
                problem_input = transformation_function(problem)
                remapping = {}
        else:
            problem_input = problem
            remapping = {}
            
        # Construct Prompt (Simulating Chat Template if needed, or raw)
        # rStar2-Agent likely uses a specific template. 
        # Using a generic chat template approximation or the model's tokenizer apply_chat_template if available.
        # vLLM `llm.chat` is newer, but let's stick to generate with formatted string for safety or check docs.
        # Assuming we can just pass the list of messages if we use the chat method? 
        # Actually vLLM's `generate` takes prompts. We need to format them.
        # For simplicity, let's assume a standard ChatML or similar format, OR use the tokenizer.
        # Better: vLLM usually handles `apply_chat_template` via tokenizer.
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_input}
        ]
        
        # We will let the tokenizer format this.
        # But we need access to the tokenizer. `llm.get_tokenizer()`
        tokenizer = llm.get_tokenizer()
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        prompts.append(formatted_prompt)
        metadata.append({
            "id": example.get('id', i),
            "original": problem,
            "transformed": problem_input,
            "ground_truth": ground_truth,
            "remapping": remapping
        })
        
    # Batch Generate
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Save Results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = experiment_name.replace(' ', '_').lower()
    run_id = f"{safe_name}_k{k}_s{seed}_{timestamp}"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    json_file = os.path.join(results_dir, f"{run_id}.json")
    
    results = []
    correct_count = 0
    total = 0
    
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
            "transformed": meta['transformed'],
            "ground_truth": meta['ground_truth'],
            "output": generated_text,
            "extracted": extracted,
            "correct": is_correct
        }
        results.append(result_entry)
        total += 1
        
    accuracy = correct_count / total if total > 0 else 0
    print(f"\nEvaluation Complete. Accuracy: {accuracy:.2%}")
    print(f"Results: {json_file}")
    
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

