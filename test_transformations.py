import argparse
import random
from datasets import load_dataset
from transformations import (
    apply_not_not_transformation,
    apply_yot_transformation,
    apply_symbol_remapping_transformation
)

def test_not_not(examples):
    print("\n=== Testing Not Not Transformation (k=3) ===\n")
    for name, text in examples:
        print(f"--- {name} ---")
        print(f"[Original]: {text}")
        print(f"[Transformed]: {apply_not_not_transformation(text, k=3)}")
        print("")

def test_yot(examples):
    print("\n=== Testing Yot Transformation (k=3) ===\n")
    for name, text in examples:
        print(f"--- {name} ---")
        print(f"[Original]: {text}")
        print(f"[Transformed]: {apply_yot_transformation(text, k=3)}")
        print("")

def test_remapping(n_examples=3, specific_indices=None):
    print(f"\n=== Testing Symbol Remapping (AIME Dataset) ===\n")
    try:
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        
        if specific_indices:
            indices = specific_indices
        else:
            indices = random.sample(range(len(dataset)), n_examples)
        
        for idx in indices:
            example = dataset[idx]
            original = example['problem']
            print(f"--- Example ID: {idx} ---")
            print(f"[Original]:\n{original}")
            transformed = apply_symbol_remapping_transformation(original, k=3)
            print(f"\n[Transformed]:\n{transformed}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error loading dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test different transformation techniques.")
    parser.add_argument("--mode", type=str, required=True, choices=["not_not", "yot", "remapping", "all"], help="Which transformation to test")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated list of indices to test (remapping only)")
    args = parser.parse_args()

    # Hardcoded examples for Not-Not/Yot validity checks
    basic_examples = [
        ("Small", "Find the sum of three distinct positive integers."),
        ("Medium", "Let a, b, and c be real numbers such that a + b + c = 0. Determine the maximum value of the product abc."),
    ]

    if args.mode in ["not_not", "all"]:
        test_not_not(basic_examples)
        
    if args.mode in ["yot", "all"]:
        test_yot(basic_examples)
        
    if args.mode in ["remapping", "all"]:
        indices = [int(i) for i in args.indices.split(",")] if args.indices else None
        test_remapping(specific_indices=indices)

if __name__ == "__main__":
    main()
