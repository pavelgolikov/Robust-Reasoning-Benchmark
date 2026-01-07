
import json
import os
import glob

def analyze_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return {'error': str(e)}
        
    total = len(data)
    if total == 0:
        return {'total': 0, 'accuracy': 0, 'correct': 0, 'failures': 0}
        
    correct = sum(1 for x in data if x.get('correct'))
    failures = sum(1 for x in data if x.get('extracted') is None)
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total,
        'failures': failures
    }

def main():
    base_dir = "experiments"
    experiments = [
        "baseline",
        "opposites",
        "opposites_not",
        "wrappers",
        "interleaved_context",
        "interleaved_substitutions",
        "numerical_wrappers"
    ]
    
    results = {}
    
    for exp in experiments:
        res_dir = os.path.join(base_dir, exp, "results")
        # Find latest JSON
        files = glob.glob(os.path.join(res_dir, "GAIR_LIMO-v2_*.json"))
        if not files:
            results[exp] = "No results found"
            continue
            
        # Sort by modification time
        latest_file = max(files, key=os.path.getmtime)
        stats = analyze_file(latest_file)
        results[exp] = {
            'file': os.path.basename(latest_file),
            'stats': stats
        }

    # Print and Save
    output_lines = []
    output_lines.append(f"{'Experiment':<30} | {'Total':<8} | {'Accuracy':<10} | {'Failures':<8} | {'File'}")
    output_lines.append("-" * 100)
    
    for exp in experiments:
        res = results.get(exp)
        if isinstance(res, str):
            output_lines.append(f"{exp:<30} | {'-':<8} | {'-':<10} | {'-':<8} | {res}")
        else:
            stats = res['stats']
            if 'error' in stats:
                output_lines.append(f"{exp:<30} | ERROR: {stats['error']}")
            else:
                acc_str = f"{stats['accuracy']:.2%}"
                output_lines.append(f"{exp:<30} | {stats['total']:<8} | {acc_str:<10} | {stats['failures']:<8} | {res['file']}")

    report = "\n".join(output_lines)
    print(report)
    
    with open("experiments/results_summary.txt", "w") as f:
        f.write(report)
        
if __name__ == "__main__":
    main()
