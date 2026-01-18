
import json
import argparse
import os
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import numpy as np

def normalize_text(text: str) -> str:
    """Basic normalization: remove latex, extra whitespace."""
    if not text:
        return ""
    # Remove latex command structure like \boxed{...} -> ...
    # This is a simple heuristic
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
    # Remove common latex symbols if they are just noise
    text = text.replace('$', '').replace('\\', '')
    return " ".join(text.split())

def make_windows(tokens: List[str], window_size: int, step_size: int = 10) -> List[str]:
    """Create sliding windows of text from tokens."""
    windows = []
    if not tokens:
        return []
    if len(tokens) <= window_size:
        return [" ".join(tokens)]
    
    for i in range(0, len(tokens) - window_size + 1, step_size):
        window = tokens[i : i + window_size]
        windows.append(" ".join(window))
    
    # Ensure the last bit is covered
    if len(tokens) > window_size:
        last_window = tokens[-window_size:]
        windows.append(" ".join(last_window))
        
    return list(set(windows)) # Remove potential duplicates

def analyze(args):
    result_file = args.file
    if not os.path.exists(result_file):
        print(f"File not found: {result_file}")
        return

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(args.model_name)
    
    with open(result_file, 'r') as f:
        data = json.load(f)

    total = len(data)
    original_correct = 0
    semantic_correct = 0
    recovered_cases = []

    print(f"Analyzing {total} samples from {result_file}...")

    # Pre-compute embeddings for efficiency if we were doing batch, 
    # but here we do per-sample because windows vary.
    
    for entry in data:
        # Original status
        is_orig_correct = entry.get('correct', False)
        if is_orig_correct:
            original_correct += 1
            semantic_correct += 1
            continue # Already correct, skip expense
            
        # For decoding task, we want the DECODED PROBLEM STATEMENT.
        # 'unmodified_original' is the ground truth original problem statement.
        target_text = entry.get('unmodified_original', '')
        model_output = entry.get('output', '')
        
        norm_target = normalize_text(target_text)
        target_tokens = norm_target.split()
        target_len = len(target_tokens)
        
        norm_output = normalize_text(model_output)
        output_tokens = norm_output.split()
        
        if not norm_target or not norm_output:
            continue

        # Window size: +/- margin logic is good, but simple "target_len" is fine
        # We can try a few window sizes around target len
        # E.g. 80%, 100%, 120% of target length to catching missing/extra words
        window_sizes = [int(target_len * 0.8), target_len, int(target_len * 1.2)]
        
        all_windows = []
        for w_size in window_sizes:
            if w_size < 1: w_size = 1
            all_windows.extend(make_windows(output_tokens, window_size=w_size, step_size=args.step_size))
        
        if not all_windows:
            continue
            
        # Encode
        target_embedding = model.encode(norm_target, convert_to_tensor=True)
        window_embeddings = model.encode(all_windows, convert_to_tensor=True)
        
        # Calculate Cosine Similarity
        # util.cos_sim returns query x corpus matrix
        cosine_scores = util.cos_sim(target_embedding, window_embeddings)[0]
        
        best_idx = int(cosine_scores.argmax())
        max_score = float(cosine_scores[best_idx])
        
        if max_score >= args.threshold:
            semantic_correct += 1
            recovered_cases.append({
                "id": entry.get('id'),
                "score": max_score,
                "target": norm_target[:100] + "...",
                "best_window": all_windows[best_idx]
            })

    print("\n" + "="*50)
    print("SEMANTIC ANALYSIS REPORT")
    print("="*50)
    print(f"File: {result_file}")
    print(f"Total Samples: {total}")
    print(f"Original Strict Accuracy: {original_correct/total:.2%} ({original_correct}/{total})")
    print(f"New Semantic Accuracy:    {semantic_correct/total:.2%} ({semantic_correct}/{total})")
    print(f"Recovered Cases:          {len(recovered_cases)}")
    
    if args.verbose and recovered_cases:
        print("\nTop 5 Recovered Cases:")
        # Sort by score desc
        recovered_cases.sort(key=lambda x: x['score'], reverse=True)
        for rc in recovered_cases[:5]:
            print("-" * 30)
            print(f"ID: {rc['id']} | Score: {rc['score']:.4f}")
            print(f"Goal:   {rc['target']}")
            print(f"Found:  {rc['best_window']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    analyze(args)
