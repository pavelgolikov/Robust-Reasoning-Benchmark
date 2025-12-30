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
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1-0528:free"

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
        "max_tokens": 8000  # Try to force a larger context window
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None

def extract_answer(text):
    """
    Extracts the answer from the model's response.
    Prioritizes \boxed{...} format.
    Fallbacks to looking for the last number.
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

    # Pattern 3: Fallback to last number
    # AIME answers are integers (0-999).
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return numbers[-1]
        
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
    return "".join(filter(str.isdigit, str(ans)))

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
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem efficiently. Do not double-check your work. Output the final answer immediately inside \\boxed{}."},
            {"role": "user", "content": problem_input}
        ]
        
        print(f"Processing Problem {i}...")
        response_data = query_openrouter(messages)
        
        if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
            model_output = response_data['choices'][0]['message']['content']
            
            # DeepSeek R1 on OpenRouter often puts reasoning in a separate field
            # and may leave content empty if it didn't reach a 'final' output separate from reasoning
            msg = response_data['choices'][0]['message']
            reasoning = msg.get('reasoning', '')
            
            if not model_output and reasoning:
                model_output = reasoning
            elif model_output and reasoning:
                # Combine them if both exist, to ensure we catch the answer if it's in content
                model_output = reasoning + "\n\n" + model_output
                
            # print(f"DEBUG: Model Output Length: {len(model_output)}")
            extracted = extract_answer(model_output)

            
            norm_extracted = normalize_answer(extracted)
            norm_truth = normalize_answer(ground_truth)
            
            is_correct = (norm_extracted == norm_truth)
            if is_correct:
                correct += 1
            
            total += 1
            
            results.append({
                "id": example.get('id', i),
                "original_problem": problem,
                "perturbed_problem": problem_input,
                "model_output": model_output,
                "extracted": extracted,
                "ground_truth": ground_truth,
                "correct": is_correct
            })
            
            print(f"  Result: {'CORRECT' if is_correct else 'INCORRECT'} (Got: {extracted}, Expected: {ground_truth})")
        else:
            print("  Failed to get response.")
            
    accuracy = correct / total if total > 0 else 0
    print(f"\nEvaluation Complete.")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--perturbation", type=str, default="none", choices=["none", "not_not"], help="Perturbation type")
    parser.add_argument("--k", type=int, default=3, help="Parameter k for not_not perturbation")
    
    args = parser.parse_args()
    
    evaluate(limit=args.limit, perturbation_type=args.perturbation if args.perturbation != "none" else None, k=args.k)
