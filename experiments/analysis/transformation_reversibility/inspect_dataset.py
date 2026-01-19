
import argparse
from datasets import load_dataset
import sys

def inspect_sample(dataset_name, split, index):
    print(f"Loading {dataset_name} ({split})...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    try:
        sample = ds[index]
    except IndexError:
        print(f"Index {index} out of range (Dataset size: {len(ds)})")
        return

    print(f"\n--- Sample {index} ---")
    for key, value in sample.items():
        print(f"\n[{key}]:")
        print(value)
    print(f"\n--- End Sample {index} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a specific sample in a dataset")
    parser.add_argument("--dataset", type=str, default="KbsdJames/Omni-MATH", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--index", type=int, required=True, help="Sample index to inspect")
    
    args = parser.parse_args()
    
    inspect_sample(args.dataset, args.split, args.index)
