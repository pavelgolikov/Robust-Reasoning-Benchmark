import os
from experiments.analysis.visualize_combined import load_metrics_data

for dataset in ["HuggingFaceH4_aime_2024", "MathArena_aime_2025"]:
    data = load_metrics_data("experiments", dataset)
    zeros = set()
    for t, t_data in data.items():
        for m, m_data in t_data.items():
            if m_data.get('length', 0) == 0:
                zeros.add(t)
    print(f"Dataset {dataset} has 0 lengths for techniques: {zeros}")
