import argparse
import json
import re

def main():
    parser = argparse.ArgumentParser(description="Test Boundary Heuristic")
    parser.add_argument("--json_file", type=str, default="/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/compound/results/Qwen_Qwen3-30B-A3B-Thinking-2507/MathArena_aime_2025/Qwen_Qwen3-30B-A3B-Thinking-2507_MathArena_aime_2025_compound_s42_20260330_155901.json", help="Path to compound JSON result file")
    args = parser.parse_args()
    
    print(f"Loading data from {args.json_file}")
    with open(args.json_file, 'r') as f:
        data = json.load(f)
        
    entries = [item for item in data if isinstance(item, dict) and "output" in item]
    print(f"Found {len(entries)} output samples.\n")
    
    pattern = re.compile(r"Problem\s*\d+", re.IGNORECASE)
    
    correct_heuristic_count = 0
    
    for i, entry in enumerate(entries):
        original = entry.get("original", "")
        output = entry.get("output", "")
        full_text = original + output
        
        prompt_problems = re.findall(r"Problem \d+:", original)
        num_distractors = max(0, len(prompt_problems) - 1)
        target_problem_num = num_distractors + 1
        
        matches = list(pattern.finditer(full_text))
        
        print(f"--- Sample {i} ---")
        print(f"Target Problem from Prompt: Problem {target_problem_num}")
        
        longest_idx = -1
        longest_len = -1
        selected_marker = None
        
        if not matches:
            print("  No 'Problem N' markers found in text.")
        else:
            groups = []
            current_group = None
            
            for j, match in enumerate(matches):
                start = match.start()
                end = matches[j+1].start() if j+1 < len(matches) else len(full_text)
                length = end - start
                
                marker_str = match.group()
                num_match = re.search(r"\d+", marker_str)
                prob_num = int(num_match.group()) if num_match else -1
                
                print(f"  Marker: '{marker_str}' at {start} | Length: {length}")
                
                if current_group is None:
                    current_group = {"prob_num": prob_num, "marker": marker_str, "total_len": length, "start_idx": start}
                elif current_group["prob_num"] == prob_num:
                    current_group["total_len"] += length
                else:
                    groups.append(current_group)
                    current_group = {"prob_num": prob_num, "marker": marker_str, "total_len": length, "start_idx": start}
            
            if current_group is not None:
                groups.append(current_group)
                
            # Filter groups to ONLY the target problem
            target_groups = [g for g in groups if g["prob_num"] == target_problem_num]
            
            if not target_groups:
                print(f"  *** MISMATCH! No groups found for Target Problem {target_problem_num} ***")
            else:
                # Find longest group among the target problem groups
                longest_group = max(target_groups, key=lambda g: g["total_len"])
                
                print(f"  -> Heuristic selects group starting with: {longest_group['marker']} (Total Length: {longest_group['total_len']}, Start: {longest_group['start_idx']})")
                
                if longest_group["prob_num"] == target_problem_num:
                    correct_heuristic_count += 1
                else:
                    print(f"  *** MISMATCH! Heuristic chose {longest_group['marker']}, but Target is Problem {target_problem_num} ***")
                
        print()
        
    if entries:
        accuracy = (correct_heuristic_count / len(entries)) * 100
        print(f"Heuristic Accuracy: {correct_heuristic_count}/{len(entries)} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()
