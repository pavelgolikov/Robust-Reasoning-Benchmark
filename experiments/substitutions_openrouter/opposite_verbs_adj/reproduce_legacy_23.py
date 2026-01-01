
import argparse
import random
import os
import json
import requests
import spacy
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found.")

# LEGACY ANTONYM DICTIONARY FROM COMMIT dd1581d6c
ANTONYM_DICTIONARY = {
    "add": "subtract", "subtract": "add",
    "plus": "minus", "minus": "plus",
    "positive": "negative", "negative": "positive",
    "multiply": "divide", "divide": "multiply",
    "increase": "decrease", "decrease": "increase",
    "maximum": "minimum", "minimum": "maximum",
    "maximize": "minimize", "minimize": "maximize",
    "least": "greatest", "greatest": "least",
    "smallest": "largest", "largest": "smallest",
    "upper": "lower", "lower": "upper",
    "top": "bottom", "bottom": "top",
    "left": "right", "right": "left",
    "first": "last", "last": "first",
    "start": "end", "end": "start",
    "begin": "finish", "finish": "begin",
    "open": "closed", "closed": "open",
    "true": "false", "false": "true",
    "same": "different", "different": "same",
    "equal": "unequal", "unequal": "equal",
    "real": "imaginary", "imaginary": "real",
    "rational": "irrational", "irrational": "rational",
    "prime": "composite", "composite": "prime",
    "odd": "even", "even": "odd",
    "sum": "difference", "difference": "sum",
    "product": "quotient", "quotient": "product",
    "numerator": "denominator", "denominator": "numerator",
    "win": "lose", "lose": "win",
    "winner": "loser", "loser": "winner",
    "best": "worst", "worst": "best",
    "success": "failure", "failure": "success",
    "rise": "fall", "fall": "rise",
    "ascend": "descend", "descend": "ascend",
    "enter": "exit", "exit": "enter",
    "include": "exclude", "exclude": "include",
    "interior": "exterior", "exterior": "interior",
    "convex": "concave", "concave": "convex",
    "finite": "infinite", "infinite": "finite",
    "converge": "diverge", "diverge": "converge",
    "constant": "variable", "variable": "constant",
    "horizontal": "vertical", "vertical": "horizontal",
    "parallel": "perpendicular", "perpendicular": "parallel",
    "similar": "dissimilar", "dissimilar": "similar",
    "congruent": "incongruent", "incongruent": "congruent"
}

# LEGACY TRANSFORMATION LOGIC FROM COMMIT dd1581d6c
def apply_opposite_semantic_remapping(text, k=2, seed=None):
    if seed is not None:
        random.seed(seed)
        
    doc = nlp(text)
    
    candidates = []
    for token in doc:
        if token.pos_ in ["VERB", "ADJ"] and token.is_alpha:
            candidates.append(token)
            
    if not candidates:
        return text
        
    num_to_swap = max(1, len(candidates) // k)
    to_swap = random.sample(candidates, num_to_swap)
    
    definitions = []
    replacements = {}
    
    for token in to_swap:
        word_lower = token.text.lower()
        original_text = token.text
        
        if word_lower in ANTONYM_DICTIONARY:
            replacement_base = ANTONYM_DICTIONARY[word_lower]
        else:
            replacement_base = "anti" + word_lower
            
        if original_text[0].isupper():
            replacement = replacement_base.capitalize()
        else:
            replacement = replacement_base
            
        replacements[token.i] = replacement
        
        def_str = f'let "{replacement}" mean "{original_text}"'
        if def_str not in definitions:
            definitions.append(def_str)
            
    output_tokens = []
    for i, token in enumerate(doc):
        if i in replacements:
            output_tokens.append(replacements[i] + token.whitespace_)
        else:
            output_tokens.append(token.text_with_ws)
            
    transformed_text = "".join(output_tokens)
    
    if definitions:
        def_block = "defyn{" + ", ".join(definitions) + "}.\n\n"
    else:
        def_block = ""
        
    return def_block + transformed_text

def evaluate(limit=None, k=1, seed=42):
    # Initialize random seed
    if seed is not None:
        random.seed(seed)
        
    # Load Dataset
    print("Loading dataset...")
    # Using local dataset loading if possible or minimal set
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/aime_2024")
    dataset = ds['train']
    
    if limit:
        dataset = dataset.select(range(limit))
        
    print(f"Starting Legacy Reproduction on {len(dataset)} examples...")

    results = []
    correct_count = 0
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    for i, example in enumerate(dataset):
        problem = example['problem']
        ground_truth = example['answer']
        
        # Apply Transformation (Legacy Logic)
        problem_input = apply_opposite_semantic_remapping(problem, k=k, seed=seed)
        
        print(f"\n--- Problem {i} ---")
        
        # LEGACY SYSTEM PROMPT
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem accurately. Output the final answer inside \\boxed{}. Each user query can be accompanied by word re-mappings. Definitions for these re-mappings will be enclosed in the 'defyn{}' block at the beginning of the user query."},
            {"role": "user", "content": problem_input}
        ]
        
        try:
            # Using the USER CONFIRMED model nex-n1, and legacy Params (temp=0)
            response = requests.post(API_URL, headers=HEADERS, json={
                "model": "nex-agi/deepseek-v3.1-nex-n1:free", 
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 16000
            })
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data:
                    content = data['choices'][0]['message']['content']
                    
                    # LEGACY EXTRACTION LOGIC
                    match = re.search(r'\\boxed\{(.*?)\}', content)
                    if match:
                        extracted_answer = match.group(1).strip()
                    else:
                        match = re.search(r'final answer is \s*(\d+)', content, re.IGNORECASE)
                        extracted_answer = match.group(1).strip() if match else "Error"

                    # Normalize for checking
                    # Legacy normalization was minimal (str comparison)
                    
                    # Actually, check legacy: "is_correct = (extracted_answer == str(ground_truth))"
                    # Yes, strict string equality.
                    is_correct = (str(extracted_answer) == str(ground_truth))
                    if is_correct:
                        correct_count += 1
                    
                    print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'} (Extracted: {extracted_answer}, Truth: {ground_truth})")
                else:
                    print(f"API Empty Choice")

            else:
                print(f"API Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error processing problem {i}: {e}")
            
    accuracy = (correct_count / len(dataset)) * 100
    print(f"\nLegacy Reproduction Complete.")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(dataset)})")
    
if __name__ == "__main__":
    evaluate(limit=30, k=1, seed=42)
