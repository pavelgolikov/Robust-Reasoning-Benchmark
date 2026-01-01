import os
import json
import re
import argparse
import time
import requests
import random
from dotenv import load_dotenv
from datasets import load_dataset
from transformations import apply_symbol_remapping_transformation

# Load environment variables
load_dotenv()
# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "nex-agi/deepseek-v3.1-nex-n1:free"

def query_openrouter(messages, retries=3):
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
    
    success = False
    response_json = None
    
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            response_json = response.json()
            success = True
            break
        except requests.exceptions.RequestException as e:
            print(f"API Request failed (attempt {attempt+1}/{retries}): {e}")
            if hasattr(e.response, 'text'):
                print(f"  Error details: {e.response.text}")
            
            if hasattr(e.response, 'status_code') and e.response.status_code == 429:
                print("  Rate limit hit (429). Waiting 30 seconds...")
                time.sleep(30)
            elif attempt < retries - 1:
                time.sleep(2)
    
    if not success or not response_json:
        print("Failed to get response after retries.")
        return None
        
    return response_json

def extract_answer(text):
    if not text:
        return None
        
    boxed_pattern = r"\\boxed\s*\{([^}]+)\}"
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()
        
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

def evaluate(limit=None, transformation_type=None, k=3):
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        
    correct = 0
    total = 0
    results = []
    
    print(f"Starting evaluation on {len(dataset)} examples. Transformation: {transformation_type}")
    
    for i, example in enumerate(dataset):
        problem = example['problem']
        ground_truth = example['answer']
        
        if transformation_type == "remapping":
            problem_input = apply_symbol_remapping_transformation(problem, k=k)
        else:
            problem_input = problem
            
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem accurately. Output the final answer inside \\boxed{}. Each user query can be accompanied by word re-mappings. Definitions for these re-mappings will be enclosed in the 'defyn{}' block at the beginning of the user query."},
            {"role": "user", "content": problem_input}
        ]
        
        print(f"Processing Problem {i}...")
        response_data = query_openrouter(messages)
        
        if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            finish_reason = choice.get('finish_reason', 'unknown')
            msg = choice['message']
            content = msg.get('content', '')
            reasoning = msg.get('reasoning', '')
            
            model_output = ""
            if reasoning:
                model_output += f"[REASONING]\n{reasoning}\n[/REASONING]\n"
            if content:
                model_output += f"[CONTENT]\n{content}\n[/CONTENT]"
            
            if finish_reason == "length":
                print(f"  WARNING: Response TRUNCATED (finish_reason='length'). extraction might fail.")
            
            text_to_search = content if content else reasoning
            extracted = extract_answer(text_to_search)
            
            norm_extracted = normalize_answer(extracted)
            norm_truth = normalize_answer(ground_truth)
            
            is_correct = (norm_extracted == norm_truth) and (norm_extracted != "")
            if is_correct:
                correct += 1
            
            total += 1
            
            # Print momentary status and reasoning snippet
            log_status = "CORRECT" if is_correct else "INCORRECT"
            if extracted is None:
                log_status = "NO_ANSWER_FOUND"
                
            print(f"  Result: {log_status} (Got: {extracted}, Expected: {ground_truth})")
            if reasoning:
                print(f"  Reasoning Snippet: {reasoning[:200].replace(chr(10), ' ')}...")
            
            results.append({
                "id": example.get('id', i),
                "original_problem": problem,
                "perturbed_problem": problem_input,
                "model_output": model_output if model_output else "EMPTY", # Full output
                "reasoning": reasoning, # Explicit reasoning field
                "extracted": extracted,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "finish_reason": finish_reason
            })
        else:
            print("  Failed to get response.")

        time.sleep(10)
            
    accuracy = correct / total if total > 0 else 0
    print(f"\nEvaluation Complete.")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    if transformation_type:
        output_file = os.path.join(output_dir, f"evaluation_{transformation_type}.json")
    else:
        output_file = os.path.join(output_dir, "evaluation_baseline.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
    
    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--transformation", type=str, default="remapping", choices=["none", "not_not", "remapping"], help="Transformation type")
    parser.add_argument("--k", type=int, default=3, help="Parameter k")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    evaluate(limit=args.limit, transformation_type=args.transformation if args.transformation != "none" else None, k=args.k)
