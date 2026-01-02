import json
import os
import re

result_file = "baseline/results/GAIR_LIMO_baseline_s42_20260102_013935.json"

def analyze():
    if not os.path.exists(result_file):
        print(f"File not found: {result_file}")
        return

    with open(result_file, 'r') as f:
        data = json.load(f)

    total = len(data)
    truncated_suspects = 0
    extraction_failures = 0
    max_len_found = 0
    limit = 12768

    print(f"Analyzing {total} samples...")

    for entry in data:
        output = entry.get('output', '')
        extracted = entry.get('extracted')
        length = len(output)
        max_len_found = max(max_len_found, length)

        if extracted is None:
            extraction_failures += 1
            # Detailed check for this failure
            print(f"-- Failure Sample {entry.get('id')} (Idx {entry.get('sample_idx')}) --")
            print(f"Length: {length}")
            print(f"Last 50 chars: {repr(output[-50:])}")
            
            # Check for unclosed boxed
            boxed_matches = list(re.finditer(r"\\boxed\{", output))
            if boxed_matches:
                last_boxed = boxed_matches[-1]
                if "}" not in output[last_boxed.end():]:
                    print("  -> TRUNCATED (Unclosed \\boxed)")
                    truncated_suspects += 1
                else:
                     print("  -> Boxed appears closed, but extraction failed.")
            else:
                 print("  -> No \\boxed found.")
                 # Check if it ends abruptly
                 if len(output) > 0 and output.strip()[-1] not in ['.', '!', '?', '}', '>', ']']:
                      print("  -> TRUNCATED (Ends abruptly)")
                      truncated_suspects += 1


    print("\nSummary:")
    print(f"Total Samples: {total}")
    print(f"Max Output Length Found: {max_len_found}")
    print(f"Extraction Failures: {extraction_failures}")
    print(f"Truncation Suspects (Unclosed boxed or Length Limit): {truncated_suspects}")

if __name__ == "__main__":
    analyze()
