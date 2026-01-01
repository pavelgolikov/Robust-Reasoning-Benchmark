import sys
import os
import argparse

# Add project root to sys.path to find 'utils' (3 levels up)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from utils.evaluation import run_evaluation
from transformation import apply_opposite_and_number_transformation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=1, help="Default k=1 (100% replacement)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        experiment_name="Opposites + Number Confusion",
        transformation_function=apply_opposite_and_number_transformation,
        system_prompt="You are a helpful math assistant. Solve the problem accurately. Output the final answer inside \\boxed{}. Each user query can be accompanied by word and number re-mappings. Definitions for these re-mappings will be enclosed in the 'defyn{}' block at the beginning of the user query.",
        results_dir=os.path.join(os.path.dirname(__file__), "results"),
        logs_dir=os.path.join(os.path.dirname(__file__), "logs"),
        limit=args.limit,
        k=args.k,
        seed=args.seed
    )
