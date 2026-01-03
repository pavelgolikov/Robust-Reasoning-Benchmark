import sys
import os
from datasets import load_dataset
import random

# Add substitutions directory to path
sys.path.append('/home/golikovp/Antigravity/Linguistic_traps/experiments/substitutions')

from wrappers.transformation import apply_wrapper_transformation

def main():
    # Load dataset
    print("Loading AIME 2024 dataset...")
    try:
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Select 3 random examples
    indices = [0, 1, 2] 
    
    # System Prompt (as currently in evaluate.py)
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    
    output_file = "wrappers_examples.txt"
    
    with open(output_file, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"System Prompt: \"{system_prompt}\"\n")
        f.write("="*80 + "\n\n")

        for i in indices:
            example = dataset[i]
            original = example['problem']
            
            # Apply transformation with k=2
            transformed = apply_wrapper_transformation(original, k=2) 
            
            f.write(f"--- Problem {i+1} ---\n")
            f.write("ORIGINAL:\n")
            f.write(original + "\n")
            f.write("\nCONVERTED:\n")
            f.write(transformed + "\n")
            f.write("\n" + "-"*80 + "\n\n")

    print(f"Examples written to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
