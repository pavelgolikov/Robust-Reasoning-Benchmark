import sys
import os
import argparse

# Add project root to sys.path to find 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from utils.evaluation import run_evaluation
from transformation import apply_not_not_transformation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        experiment_name="Not Not",
        transformation_function=apply_not_not_transformation,
        system_prompt="You are a helpful math assistant. Solve the problem accurately. Output the final answer inside \\boxed{}.",
        results_dir=os.path.join(os.path.dirname(__file__), "results"),
        logs_dir=os.path.join(os.path.dirname(__file__), "logs"),
        limit=args.limit,
        k=args.k,
        seed=args.seed
    )
