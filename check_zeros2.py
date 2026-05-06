import os
from experiments.analysis.visualize_combined import load_metrics_data

for dataset in ["HuggingFaceH4_aime_2024", "MathArena_aime_2025"]:
    data = load_metrics_data("experiments", dataset)
    print(f"\n--- {dataset} ---")
    for t, t_data in data.items():
        if t not in ['baseline', 'snake_horizontal', 'rail_fence']:
            continue
        for m, m_data in t_data.items():
            if m_data.get('length', 0) == 0:
                print(f"Zero length: tech {t}, model {m}")
