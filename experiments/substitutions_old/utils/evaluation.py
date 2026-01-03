import os
import json
import re
import time
import requests
import random
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env")))

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "nex-agi/deepseek-v3.1-nex-n1:free"

def query_openrouter(messages, retries=5):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 32000
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            if response.status_code == 429:
                print(f"  Rate limit hit (429). Retrying in {2**attempt}s...")
                time.sleep(2**attempt)
                continue
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API Request failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
                
    return None

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

def run_evaluation(
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
    Generic evaluation loop.
    Generates timestamped log and result files to ensure reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        
    print(f"Starting {experiment_name} on {len(dataset)} examples. Seed={seed}, k={k}")
    
    # Generate unique run ID / filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = experiment_name.replace(' ', '_').lower()
    run_id = f"{safe_name}_k{k}_s{seed}_{timestamp}"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, f"{run_id}.log")
    json_file = os.path.join(results_dir, f"{run_id}.json")
    
    results = []
    correct_count = 0
    total = 0
    
    with open(log_file, "w") as log:
        log.write(f"Evaluation Started: {time.ctime()}\n")
        log.write(f"Experiment: {experiment_name}\n")
        log.write(f"Run ID: {run_id}\n")
        log.write(f"Parameters: k={k}, seed={seed}, limit={limit}\n\n")

    for i, example in enumerate(dataset):
        problem = example['problem']
        ground_truth = example['answer']
        
        # Apply transformation if provided
        if transformation_function:
            try:
                problem_input = transformation_function(problem, k=k, seed=seed)
            except TypeError:
                try:
                    problem_input = transformation_function(problem, k=k)
                except TypeError:
                    problem_input = transformation_function(problem) # No k
        else:
            problem_input = problem
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_input}
        ]
        
        print(f"Processing Problem {i}...")
        response_data = query_openrouter(messages)
        
        result_entry = {
            "id": example.get('id', i),
            "original": problem,
            "transformed": problem_input,
            "ground_truth": ground_truth,
            "output": "",
            "extracted": None,
            "correct": False
        }
        
        if response_data and 'choices' in response_data:
            content = response_data['choices'][0]['message']['content'] or ""
            reasoning = response_data['choices'][0]['message'].get('reasoning', "")
            
            full_text = f"[REASONING]\n{reasoning}\n[CONTENT]\n{content}"
            result_entry["output"] = full_text
            
            # Extract
            extracted = extract_answer(content if content else reasoning)
            result_entry["extracted"] = extracted
            
            # Normalize and Check
            if normalize_answer(extracted) == normalize_answer(ground_truth) and extracted:
                result_entry["correct"] = True
                correct_count += 1
                
            status = "CORRECT" if result_entry["correct"] else "INCORRECT"
            print(f"  Result: {status} (Got: {extracted}, Expected: {ground_truth})")
            
            # Log to file
            with open(log_file, "a") as log:
                log.write(f"--- Problem {i} ---\n")
                log.write(f"Result: {status}\n")
                log.write(f"Extracted: {extracted}, Truth: {ground_truth}\n\n")
        else:
            print("  Failed to get response.")
            with open(log_file, "a") as log:
                log.write(f"--- Problem {i} ---\nAPI FAILURE\n\n")
                
        results.append(result_entry)
        total += 1
        
        # Rate limit pause
        time.sleep(5)
        
    accuracy = correct_count / total if total > 0 else 0
    print(f"\nEvaluation Complete. Accuracy: {accuracy:.2%}")
    print(f"Logs: {log_file}")
    print(f"Results: {json_file}")
    
    # Save Results JSON
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

