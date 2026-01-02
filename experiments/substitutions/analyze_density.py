import json
import re

result_file = "baseline/results/GAIR_LIMO_baseline_s42_20260102_013935.json"

def is_truncated_heuristic(output):
    if len(output) == 0: return True
    last_chars = output[-20:].strip()
    if last_chars and last_chars[-1] not in ['.', '!', '?', '}', '>', ']']:
        return True
    # unclosed boxed
    boxed_matches = list(re.finditer(r"\\boxed\{", output))
    if boxed_matches:
        last_boxed = boxed_matches[-1]
        if "}" not in output[last_boxed.end():]:
            return True
    return False

def analyze_density():
    with open(result_file, 'r') as f:
        data = json.load(f)

    truncated_stats = []
    successful_stats = []

    for entry in data:
        output = entry.get('output', '')
        length = len(output)
        
        # Simple heuristic for token density:
        # Count non-ascii characters (Chinese, etc usually) which are token-heavy
        non_ascii = len(re.sub(r'[\x00-\x7F]+', '', output))
        non_ascii_ratio = non_ascii / length if length > 0 else 0
        
        stat = {'len': length, 'non_ascii': non_ascii, 'ratio': non_ascii_ratio, 'id': entry['id']}
        
        if is_truncated_heuristic(output):
            truncated_stats.append(stat)
        else:
            successful_stats.append(stat)

    # Sort by length
    truncated_stats.sort(key=lambda x: x['len'])
    successful_stats.sort(key=lambda x: x['len'])

    print(f"--- Truncated Samples (Total {len(truncated_stats)}) ---")
    print(f"{'ID':<5} | {'Length':<10} | {'Non-ASCII %':<12}")
    for s in truncated_stats:
        print(f"{s['id']:<5} | {s['len']:<10} | {s['ratio']:.2%}")

    print(f"\n--- Top 10 Longest Successful Samples ---")
    print(f"{'ID':<5} | {'Length':<10} | {'Non-ASCII %':<12}")
    for s in successful_stats[-10:]:
        print(f"{s['id']:<5} | {s['len']:<10} | {s['ratio']:.2%}")

if __name__ == "__main__":
    analyze_density()
