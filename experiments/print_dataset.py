import argparse
import os
import sys
from datasets import load_dataset
from util import remove_latex_comments

def main():
    parser = argparse.ArgumentParser(description="Print dataset questions in plain text.")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--sample_range", type=str, default=None, help="Range of sample indices to process, e.g. '0-10' or '5' or '1,3,5'")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}...")
    try:
        dataset = load_dataset(args.dataset, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    indices = list(range(len(dataset)))
    
    if args.sample_range:
        indices = []
        try:
            parts = args.sample_range.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start, end)) # Python range exclusive
                else:
                    indices.append(int(part))
        except ValueError:
            print("Error: Invalid sample_range format.")
            return

        # Validate indices
        valid_indices = [i for i in indices if 0 <= i < len(dataset)]
        if len(valid_indices) != len(indices):
            print(f"Warning: Some indices were out of range (0-{len(dataset)-1}). Kept {len(valid_indices)} valid indices.")
        indices = valid_indices
        
        if not indices:
             print("Error: No valid indices found in sample_range.")
             return

    # Select the subset based on indices to iterate easily or just iterate directly
    subset = dataset.select(indices)
    
    print(f"Printing {len(subset)} questions:\n")
    
    for i, example in enumerate(subset):
        original_idx = indices[i]
        problem_text = example.get('problem', example.get('question', '')) # Fallback for other datasets
        
        # Optionally clean comments if desired, or just raw?
        # User asked for "plain text format". Let's provide raw text but denote ID.
        print(f"--- Question ID: {example.get('id', original_idx)} (Index: {original_idx}) ---")
        print(problem_text)
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
