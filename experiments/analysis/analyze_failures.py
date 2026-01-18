
import json
import argparse
import os
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

def normalize_text(text: str) -> str:
    """Basic normalization: remove latex, extra whitespace."""
    if not text:
        return ""
    # Remove latex command structure like \boxed{...} -> ...
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer model on {device}...")
    model = SentenceTransformer(args.model_name, device=device)
    
    with open(result_file, 'r') as f:
        data = json.load(f)

    total = len(data)
    results_summary = {
        "source_file": result_file,
        "total_samples": total,
        "original_correct": 0,
        "semantic_correct": 0,
        "recovered_cases": []
    }

    print(f"Analyzing {total} samples from {result_file}...")

    # Iterate with progress bar
    for entry in tqdm(data, desc="Processing"):
        is_orig_correct = entry.get('correct', False)
        if is_orig_correct:
            results_summary["original_correct"] += 1
            results_summary["semantic_correct"] += 1
            continue 
            
        target_text = entry.get('unmodified_original', '')
        model_output = entry.get('output', '')
        
        norm_target = normalize_text(target_text)
        target_tokens = norm_target.split()
        target_len = len(target_tokens)
        
        norm_output = normalize_text(model_output)
        output_tokens = norm_output.split()
        
        if not norm_target or not norm_output:
            continue

        # Window sizes: 80%, 100%, 120% of target length
        window_sizes = [int(target_len * 0.8), target_len, int(target_len * 1.2)]
        
        all_windows = []
        for w_size in window_sizes:
            if w_size < 1: w_size = 1
            all_windows.extend(make_windows(output_tokens, window_size=w_size, step_size=args.step_size))
        
        if not all_windows:
            continue
            
        # Encode
        target_embedding = model.encode(norm_target, convert_to_tensor=True, show_progress_bar=False)
        window_embeddings = model.encode(all_windows, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate scores
        cosine_scores = util.cos_sim(target_embedding, window_embeddings)[0]
        
        best_idx = int(cosine_scores.argmax())
        max_score = float(cosine_scores[best_idx])
        
        if max_score >= args.threshold:
            results_summary["semantic_correct"] += 1
            results_summary["recovered_cases"].append({
                "id": entry.get('id'),
                "score": max_score,
                "target": norm_target,
                "best_window": all_windows[best_idx]
            })

    # Stats calculation
    orig_acc = results_summary["original_correct"] / total
    sem_acc = results_summary["semantic_correct"] / total
    
    results_summary["original_accuracy"] = orig_acc
    results_summary["semantic_accuracy"] = sem_acc

    print("\n" + "="*50)
    print("SEMANTIC ANALYSIS REPORT")
    print("="*50)
    print(f"Original Accuracy: {orig_acc:.2%} ({results_summary['original_correct']}/{total})")
    print(f"New Semantic Accuracy: {sem_acc:.2%} ({results_summary['semantic_correct']}/{total})")
    print(f"Recovered Cases: {len(results_summary['recovered_cases'])}")
    
    # Save to file
    if args.output_file:
        output_path = args.output_file
    else:
        # Default: append _semantic_analysis.json to input filename relative path
        base, _ = os.path.splitext(result_file)
        output_path = f"{base}_semantic_analysis.json"
        
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output_file", type=str, help="Path to save analysis results")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    
    analyze(args)
