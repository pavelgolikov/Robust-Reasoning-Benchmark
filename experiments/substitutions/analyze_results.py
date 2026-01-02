import json
import os
import re

result_file = "baseline/results/GAIR_LIMO-v2_baseline_s42_20260102_043229.json"

def analyze():
    if not os.path.exists(result_file):
        print(f"File not found: {result_file}")
        return

    with open(result_file, 'r') as f:
        data = json.load(f)

    total = len(data)
    correct = 0
    extraction_failures = 0
    max_len_found = 0
    truncated_suspects = 0

    print(f"Analyzing {total} samples from {result_file}...")

    for entry in data:
        output = entry.get('output', '')
        extracted = entry.get('extracted')
        is_correct = entry.get('correct', False)
        length = len(output)
        max_len_found = max(max_len_found, length)

        if is_correct:
            correct += 1
        
        if extracted is None:
            extraction_failures += 1
            # Check for truncation in failures
            # Unclosed \boxed
            boxed_matches = list(re.finditer(r"\\boxed\{", output))
            if boxed_matches:
                last_boxed = boxed_matches[-1]
                if "}" not in output[last_boxed.end():]:
                    truncated_suspects += 1
            elif len(output) > 0 and output.strip()[-1] not in ['.', '!', '?', '}', '>', ']']:
                truncated_suspects += 1
        
        # Check truncation for ALL samples (even successful ones can be close to limit)
        # If user used 32000 tokens ~ 120k chars.
        
    print("\nSummary:")
    print(f"Total Samples: {total}")
    print(f"Accuracy: {correct/total:.2%} ({correct}/{total})")
    print(f"Max Output Length: {max_len_found} chars")
    print(f"Extraction Failures: {extraction_failures}")
    print(f"Truncation Suspects (in Failures): {truncated_suspects}")

    # Pass@1 equivalent (aggregated by ID)
    # Group by 'id'
    by_id = {}
    for entry in data:
        eid = entry.get('id')
        if eid not in by_id: by_id[eid] = []
        by_id[eid].append(entry.get('correct'))
    
    pass_rates = [sum(v)/len(v) for v in by_id.values()]
    avg_pass_rate = sum(pass_rates) / len(pass_rates)
    print(f"Problem-Level Average Pass Rate: {avg_pass_rate:.2%}")

if __name__ == "__main__":
    analyze()
