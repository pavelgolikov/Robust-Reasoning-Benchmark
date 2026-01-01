import sys
import os
import argparse

# Add parent directory to path to find local_evaluation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from local_evaluation import run_local_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="rstar2-reproduce/rStar2-Agent-14B", help="Path to the model to evaluate")
    args = parser.parse_args()

    run_local_evaluation(
        experiment_name="Baseline (No Trap) - Killarney",
        transformation_function=None, # No transformation = Baseline
        system_prompt="You are a helpful math assistant. Solve the problem accurately. Output the final answer inside \\boxed{}. Each user query can be accompanied by word re-mappings. Definitions for these re-mappings will be enclosed in the 'defyn{}' block at the beginning of the user query.",
        results_dir=os.path.join(os.path.dirname(__file__), "results"),
        logs_dir=os.path.join(os.path.dirname(__file__), "logs"),
        k=None, # Not applicable for baseline
        seed=args.seed,
        model_path=args.model
    )
