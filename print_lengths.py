import json
import os
from experiments.analysis.visualize_combined import load_metrics_data

for dataset in ["HuggingFaceH4_aime_2024", "MathArena_aime_2025"]:
    data = load_metrics_data("experiments", dataset)
    print(f"\n--- {dataset} ---")
    
    # Just print the first model as an example, or all
    model = "gpt-5.4" # Or any
    print(f"Model: {model}")
    for t in data.keys():
        if model in data[t]:
            print(f"{t}: {data[t][model].get('length', 0)}")
