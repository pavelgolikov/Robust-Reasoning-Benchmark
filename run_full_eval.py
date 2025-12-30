import argparse
from evaluate import evaluate

def run_full_eval(limit=None, k=3):
    print("========================================")
    print("Running Baseline Evaluation (No Perturbation)")
    print("========================================")
    baseline_acc, baseline_results = evaluate(limit=limit, perturbation_type=None)
    
    print("\n")
    print("========================================")
    print(f"Running Adversarial Evaluation (Not Not, k={k})")
    print("========================================")
    adv_acc, adv_results = evaluate(limit=limit, perturbation_type="not_not", k=k)
    
    print("\n")
    print("========================================")
    import json
    import os
    
    os.makedirs("results", exist_ok=True)
    
    # Save raw results
    output_data = {
        "baseline": {
            "accuracy": baseline_acc,
            "results": baseline_results
        },
        "adversarial": {
            "perturbation": "not_not",
            "k": k,
            "accuracy": adv_acc,
            "results": adv_results
        }
    }
    
    with open("results/evaluation_data.json", "w") as f:
        json.dump(output_data, f, indent=2)
        
    # Save report
    with open("results/evaluation_report.md", "w") as f:
        f.write(f"# Linguistic Traps Evaluation Report\n\n")
        f.write(f"## Summary\n")
        f.write(f"- **Baseline Accuracy**: {baseline_acc:.2%}\n")
        f.write(f"- **Adversarial Accuracy (Not Not, k={k})**: {adv_acc:.2%}\n")
        f.write(f"- **Performance Gap**: {baseline_acc - adv_acc:.2%}\n\n")
        
        f.write(f"## Detailed Flipping Analysis\n")
        f.write(f"| ID | Baseline | Adversarial | Flip Type | Original Problem | Perturbed Problem |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        
        for b_res, a_res in zip(baseline_results, adv_results):
            if b_res['id'] == a_res['id']:
                status_b = "CORRECT" if b_res['correct'] else "INCORRECT"
                status_a = "CORRECT" if a_res['correct'] else "INCORRECT"
               
                flip_type = ""
                if status_b == "CORRECT" and status_a == "INCORRECT":
                    flip_type = "SUCCESS (Trap Worked)"
                elif status_b == "INCORRECT" and status_a == "CORRECT":
                    flip_type = "REVERSE (Trap Helped)"
                elif status_b != status_a:
                     flip_type = "CHANGED"
                
                # Sanitize newlines for table
                p_orig = b_res['original_problem'][:100].replace('\n', ' ') + "..."
                p_adv = a_res['perturbed_problem'][:100].replace('\n', ' ') + "..."
                
                if status_b != status_a:
                    f.write(f"| {b_res['id']} | {status_b} | {status_a} | {flip_type} | {p_orig} | {p_adv} |\n")

    print(f"\nResults saved to results/evaluation_data.json and results/evaluation_report.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per run")
    parser.add_argument("--k", type=int, default=3, help="Parameter k for not_not perturbation")
    args = parser.parse_args()
    
    run_full_eval(limit=args.limit, k=args.k)
