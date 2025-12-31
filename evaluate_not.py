import os
import json
import re
import argparse
import time
import requests
from dotenv import load_dotenv
from datasets import load_dataset
from perturbation import apply_not_not_perturbation

# Load environment variables
load_dotenv()
# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "nex-agi/deepseek-v3.1-nex-n1:free"
# MODEL_NAME = "deepseek/deepseek-r1-0528:free"

def query_openrouter(messages, retries=3):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        # "HTTP-Referer": "http://localhost:3000", # Optional
        # "X-Title": "Linguistic Traps Eval" # Optional
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
            
            # Rate limit handling
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
    """
    Extracts the answer from the model's response.
    Prioritizes \boxed{...} format.
    STRICTER FALLBACK: Returns None if no clear answer format is found.
    Does NOT blindly pick the last number.
    """
    if not text:
        return None
        
    # Pattern 1: Standard boxed command (most reliable)
    # Search for the *last* occurrence of \boxed{...}
    boxed_pattern = r"\\boxed\s*\{([^}]+)\}"
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()
        
    # Pattern 2: Explicit "The answer is X" or "so X"
    # Look for "The answer is <number>" or "so <number>" or "equals <number>"
    # We take the last match to capture the final conclusion.
    answer_pattern = r"(?:The answer is|result is|so|equals)\s*[:=]?\s*(\d{1,4})(?:\.|,|\s|$)"
    matches = re.findall(answer_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1]

    # STOIC CHANGE: Removed "Pattern 3: Fallback to last number"
    # This was causing high false positive rates on truncated responses.
    
    return None

def normalize_answer(ans):
    """
    Normalizes the answer string for comparison.
    AIME answers are integers.
    """
    if ans is None:
        return ""
    # Remove whitespace and potential non-numeric chars if mixed
    # Just keep digits
    digits = "".join(filter(str.isdigit, str(ans)))
    if not digits:
        return ""
    # Convert to int and back to str to remove leading zeros (e.g. "045" -> "45")
    return str(int(digits))

def evaluate(limit=None, perturbation_type=None, k=3):
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        
    correct = 0
    total = 0
    results = []
    
    print(f"Starting evaluation on {len(dataset)} examples. Perturbation: {perturbation_type}")
    
    for i, example in enumerate(dataset):
        problem = example['problem']
        ground_truth = example['answer']
        
        # Apply perturbation
        if perturbation_type == "not_not":
            problem_input = apply_not_not_perturbation(problem, k=k)
        else:
            problem_input = problem
            
        # Construct messages
        # System prompt can be simple or explicit about format
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem accurately. Output the final answer inside \\boxed{}."},
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
            
            # Combine logic
            model_output = ""
            if reasoning:
                model_output += f"[REASONING]\n{reasoning}\n[/REASONING]\n"
            if content:
                model_output += f"[CONTENT]\n{content}\n[/CONTENT]"
            
            # Check for truncation
            if finish_reason == "length":
                print(f"  WARNING: Response TRUNCATED (finish_reason='length'). extraction might fail.")
            
            # Extract from content first, then reasoning if content is empty/missing
            # Actually, extract_answer searches text. We should feed it the part where the answer is likely to be.
            # R1 usually puts answer in content after reasoning.
            
            text_to_search = content if content else reasoning
            # If we stitched content, text_to_search might be huge.
            # Searching the *end* of the string is safer for the final answer.
            
            extracted = extract_answer(text_to_search)
            
            norm_extracted = normalize_answer(extracted)
            norm_truth = normalize_answer(ground_truth)
            
            is_correct = (norm_extracted == norm_truth) and (norm_extracted != "")
            if is_correct:
                correct += 1
            
            total += 1
            
            results.append({
                "id": example.get('id', i),
                "original_problem": problem,
                "perturbed_problem": problem_input,
                "model_output": model_output[-500:] if model_output else "EMPTY", # Log last 500 chars
                "extracted": extracted,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "finish_reason": finish_reason
            })
            
            log_status = "CORRECT" if is_correct else "INCORRECT"
            if extracted is None:
                log_status = "NO_ANSWER_FOUND"
                
            print(f"  Result: {log_status} (Got: {extracted}, Expected: {ground_truth})")
        else:
            print("  Failed to get response.")

        # Rate limiting prevent
        time.sleep(10)
            
    accuracy = correct / total if total > 0 else 0
    print(f"\nEvaluation Complete.")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    # Use perturbation type for filename
    safe_perturbation_name = perturbation_type if perturbation_type else "baseline"
    output_file = os.path.join(output_dir, f"evaluation_{safe_perturbation_name}.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
    
    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--perturbation", type=str, default="none", choices=["none", "not_not"], help="Perturbation type")
    parser.add_argument("--k", type=int, default=3, help="Parameter k for not_not perturbation")
    
    args = parser.parse_args()
    
    evaluate(limit=args.limit, perturbation_type=args.perturbation if args.perturbation != "none" else None, k=args.k)
