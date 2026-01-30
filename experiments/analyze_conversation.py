
import json
import os
import sys
import statistics

# Path to latest file
result_file = "experiments/context_saturation/conv_results/tiiuae_Falcon-H1R-7B/HuggingFaceH4_aime_2024/tiiuae_Falcon-H1R-7B_HuggingFaceH4_aime_2024_s42_20260130_135449_CONVERSATION.json"

def analyze():
    if not os.path.exists(result_file):
        print(f"File not found: {result_file}")
        return

    with open(result_file, 'r') as f:
        data = json.load(f)
        
    total = len(data)
    correct = sum(1 for d in data if d.get('correct', False))
    
    print(f"Total Samples: {total}")
    print(f"Accuracy: {correct/total:.2%} ({correct}/{total})")
    
    # Token Usage Stats
    distractor_tokens = []
    solution_tokens = []
    
    distractor_counts_per_sample = [] # How many distractors were actually processed?
    
    for d in data:
        usage = d.get('token_usage', {})
        
        if 'distractors_avg_tokens' in usage:
            avg_dist = usage.get('distractors_avg_tokens', 0)
            count = usage.get('distractors_count', 0)
            sol = usage.get('solution_tokens', 0)
            
            if count > 0:
                distractor_tokens.extend([avg_dist] * int(count))
            solution_tokens.append(sol)
            distractor_counts_per_sample.append(count)
            
        elif 'distractor_avg' in usage:
             # My previous format (just in case)
            avg_dist = usage.get('distractor_avg', 0)
            count = usage.get('distractor_count', 0)
            sol = usage.get('solution', 0)
            
            if count > 0:
                distractor_tokens.extend([avg_dist] * int(count))
            solution_tokens.append(sol)
            distractor_counts_per_sample.append(count)
            
        else:
            # Old Format (detailed dict)
            current_sample_distractors = []
            for k, v in usage.items():
                if k.startswith('distractor') and not k.endswith('_tokens') and not k.endswith('_count'):
                     # Only match "distractor 1", "distractor 2" etc. 
                     # Avoid matching summary keys if they exist but fell through
                    current_sample_distractors.append(v)
                    distractor_tokens.append(v)
                elif k == 'solution':
                    solution_tokens.append(v)
                    
            distractor_counts_per_sample.append(len(current_sample_distractors))
        
    avg_dist_tokens = statistics.mean(distractor_tokens) if distractor_tokens else 0
    avg_sol_tokens = statistics.mean(solution_tokens) if solution_tokens else 0
    
    avg_dist_count = statistics.mean(distractor_counts_per_sample) if distractor_counts_per_sample else 0
    
    print("\n--- Token Usage Stats ---")
    print(f"Average Tokens per Distractor Turn: {avg_dist_tokens:.1f}")
    print(f"Average Tokens for Final Solution: {avg_sol_tokens:.1f}")
    print(f"Average Distractors Processed per Sample: {avg_dist_count:.1f}")
    
    # Check correctness breakdown
    print("\n--- Correctness Breakdown ---")
    # Maybe check if there is a correlation between token usage and correctness?
    correct_sol_tokens = [d.get('token_usage', {}).get('solution', 0) for d in data if d.get('correct', False)]
    incorrect_sol_tokens = [d.get('token_usage', {}).get('solution', 0) for d in data if not d.get('correct', False)]
    
    print(f"Avg Solution Tokens (Correct): {statistics.mean(correct_sol_tokens) if correct_sol_tokens else 0:.1f}")
    print(f"Avg Solution Tokens (Incorrect): {statistics.mean(incorrect_sol_tokens) if incorrect_sol_tokens else 0:.1f}")

if __name__ == "__main__":
    analyze()
