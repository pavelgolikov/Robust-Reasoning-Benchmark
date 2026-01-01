

import argparse
import random
import os
import json
import requests
import spacy
from transformations import apply_opposite_semantic_remapping
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Proceeding (assuming transformation handles it).")

def evaluate(limit=None, k=2, seed=42):
    """
    Evaluates the Opposite Semantic Remapping transformation.
    """
    # Initialize random seed
    if seed is not None:
        random.seed(seed)
        
    # Load Dataset
    print("Loading dataset...")
    ds = load_dataset("HuggingFaceH4/aime_2024")
    dataset = ds['train']
    
    if limit:
        dataset = dataset.select(range(limit))
        
    print(f"Starting Semantic Opposites Evaluation on {len(dataset)} examples with Seed={seed}...")
    print(f"Transformation: Opposite Semantic Remapping (k={k}, ~{100/k:.1f}% swaps)")

    results = []
    correct_count = 0
    
    # API Setup
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    for i, example in enumerate(dataset):
        problem = example['problem']
        ground_truth = example['answer']
        
        # Apply Transformation
        problem_input = apply_opposite_semantic_remapping(problem, k=k, seed=seed)
        
        print(f"\n--- Problem {i} ---")
        # print(f"Input Snippet: {problem_input[:200]}...")
        
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem accurately. Output the final answer inside \\boxed{}. Each user query can be accompanied by word re-mappings. Definitions for these re-mappings will be enclosed in the 'defyn{}' block at the beginning of the user query."},
            {"role": "user", "content": problem_input}
        ]
        
        try:
            response = requests.post(API_URL, headers=HEADERS, json={
                "model": "deepseek/deepseek-chat", # deepseek-v3.1-nex-n1 if available, ensuring consistency with previous
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 16000 # Ensure enough for reasoning
            })
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Extract Boxed Answer
                import re
                match = re.search(r'\\boxed\{(.*?)\}', content)
                if match:
                    extracted_answer = match.group(1).strip()
                else:
                    # Fallback pattern for simple numbers
                    match = re.search(r'final answer is \s*(\d+)', content, re.IGNORECASE)
                    extracted_answer = match.group(1).strip() if match else "Error"

                # Check Correctness
                is_correct = (extracted_answer == str(ground_truth))
                if is_correct:
                    correct_count += 1
                
                result_entry = {
                    "original_problem": problem,
                    "transformed_problem": problem_input,
                    "ground_truth": ground_truth,
                    "model_output": content,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct
                }
                results.append(result_entry)
                
                print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'} (Extracted: {extracted_answer}, Truth: {ground_truth})")
                
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error processing problem {i}: {e}")
            
    # Save Results
    os.makedirs('results', exist_ok=True)
    with open('results/evaluation_opposites.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    accuracy = (correct_count / len(dataset)) * 100
    print(f"\nEvaluation Complete.")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(dataset)})")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    parser.add_argument("--k", type=int, default=2, help="Frequency parameter (1/k swaps). Default 2 (50%).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    from dotenv import load_dotenv
    load_dotenv()
    
    evaluate(limit=args.limit, k=args.k, seed=args.seed)
