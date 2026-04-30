import argparse
import json
import re
import os
import torch
import random
from transformers import AutoTokenizer

def get_system_token_boundary(full_text: str, tokenizer) -> tuple[int, int]:
    """Finds the token boundary for the end of the system prompt (start of first 'Problem N'). Returns (token_idx, char_idx)."""
    pattern = re.compile(r"Problem\s*\d+", re.IGNORECASE)
    match = pattern.search(full_text)
    if not match:
        return 0, 0
        
    char_split_idx = match.start()
    
    # Tokenize with offset mapping
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    
    # Map character index to token index
    for token_idx, (start_char, end_char) in enumerate(offsets):
        if end_char > char_split_idx:
            return token_idx, char_split_idx
            
    return len(offsets), char_split_idx

def get_target_token_boundary(full_text: str, target_problem_num: int, tokenizer) -> tuple[int, int]:
    """Finds the token boundary for the longest contiguous problem-solving block. Returns (token_idx, char_idx)."""
    pattern = re.compile(r"Problem\s*\d+", re.IGNORECASE)
    matches = list(pattern.finditer(full_text))
    
    if not matches:
        raise ValueError("No 'Problem N' markers found in text.")
        
    groups = []
    current_group = None
    
    for j, match in enumerate(matches):
        start = match.start()
        end = matches[j+1].start() if j+1 < len(matches) else len(full_text)
        length = end - start
        
        marker_str = match.group()
        num_match = re.search(r"\d+", marker_str)
        prob_num = int(num_match.group()) if num_match else -1
        
        if current_group is None:
            current_group = {"prob_num": prob_num, "marker": marker_str, "total_len": length, "start_idx": start}
        elif current_group["prob_num"] == prob_num:
            current_group["total_len"] += length
        else:
            groups.append(current_group)
            current_group = {"prob_num": prob_num, "marker": marker_str, "total_len": length, "start_idx": start}
            
    if current_group is not None:
        groups.append(current_group)
        
    target_groups = [g for g in groups if g["prob_num"] == target_problem_num]
    if not target_groups:
        raise ValueError(f"No groups found for Target Problem {target_problem_num}")
        
    longest_group = max(target_groups, key=lambda g: g["total_len"])
    char_split_idx = longest_group["start_idx"]
    
    # Tokenize with offset mapping
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    
    # Map character index to token index
    for token_idx, (start_char, end_char) in enumerate(offsets):
        if end_char > char_split_idx:
            return token_idx, char_split_idx
            
    return len(offsets), char_split_idx

def get_target_statement_boundary(full_text: str, original: str, target_problem_num: int, tokenizer) -> tuple[int, int]:
    """Finds the token boundaries for the target problem statement in the input prompt."""
    
    # Find the target problem marker in original
    pattern = re.compile(rf"Problem\s*{target_problem_num}\s*:", re.IGNORECASE)
    match = pattern.search(original)
    if not match:
        raise ValueError(f"Could not find 'Problem {target_problem_num}:' in original prompt")
    
    # Find where original appears in full_text
    original_offset = full_text.find(original)
    if original_offset == -1:
        # Fallback if chat template significantly mangles whitespace or adds wrappers
        print(f"  Warning: Could not find exact original prompt text within full_text. Using heuristic match.")
        original_offset = 0 # Placeholder or more complex search
        
    # Map to full_text character coordinates
    stmt_start_char = original_offset + match.start()
    stmt_end_char = original_offset + len(original)
    
    # Tokenize with offset mapping
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    
    # Find stmt_start_token: first token that extends past stmt_start_char
    stmt_start_token = len(offsets)
    for token_idx, (start_char, end_char) in enumerate(offsets):
        if end_char > stmt_start_char:
            stmt_start_token = token_idx
            break
    
    # Find stmt_end_token: first token that starts at or after stmt_end_char
    stmt_end_token = len(offsets)
    for token_idx, (start_char, end_char) in enumerate(offsets):
        if start_char >= stmt_end_char:
            stmt_end_token = token_idx
            break
    
    return stmt_start_token, stmt_end_token

def print_context(tokens, idx, tokenizer, window=15, label="Boundary"):
    """Prints a snippet of text around a token index."""
    start = max(0, idx - window)
    end = min(len(tokens), idx + window)
    
    pre = tokenizer.decode(tokens[start:idx])
    post = tokenizer.decode(tokens[idx:end])
    
    print(f"  [ {label} ] (Token {idx})")
    print(f"    ...{pre.replace('\n', '\\n')} >>> BOUNDARY <<< {post.replace('\n', '\\n')}...")

def main():
    parser = argparse.ArgumentParser(description="Test Boundary Heuristic with Tokenizer")
    parser.add_argument("--json_file", type=str, default="/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/compound/results/Qwen_Qwen3-30B-A3B-Thinking-2507/MathArena_aime_2025/Qwen_Qwen3-30B-A3B-Thinking-2507_MathArena_aime_2025_compound_s42_20260330_155901.json", help="Path to compound JSON result file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    
    # Extract model ID from JSON file path
    safe_model_name = os.path.normpath(args.json_file).split(os.sep)[-3]
    model_id = safe_model_name.replace("_", "/", 1)
    
    print(f"Loading data from {args.json_file}")
    print(f"Model ID: {model_id}")
    
    with open(args.json_file, 'r') as f:
        data = json.load(f)
        
    entries = [item for item in data if isinstance(item, dict) and "output" in item]
    num_to_process = min(len(entries), args.num_samples)
    
    random.seed(args.seed)
    # Get random indices to process
    indices = list(range(len(entries)))
    selected_indices = random.sample(indices, num_to_process)
    
    print(f"Found {len(entries)} output samples. Testing {num_to_process} random samples (Seed: {args.seed}).\n")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    for i, idx in enumerate(selected_indices):
        entry = entries[idx]
        original = entry.get("original", "")
        system_prompt = entry.get("system_prompt", "")
        output = entry.get("output", "")
        
        # Reconstruct full text using chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = formatted_prompt + output
        
        # Determine target problem num
        prompt_problems = re.findall(r"Problem \d+:", original)
        num_distractors = max(0, len(prompt_problems) - 1)
        target_problem_num = num_distractors + 1
        
        print(f"--- Sample {i+1} (Index: {idx}, Target: Problem {target_problem_num}) ---")
        
        try:
            sys_end, _ = get_system_token_boundary(full_text, tokenizer)
            t_stmt_start, t_stmt_end = get_target_statement_boundary(full_text, original, target_problem_num, tokenizer)
            t_start, _ = get_target_token_boundary(full_text, target_problem_num, tokenizer)
            
            tokens = tokenizer.encode(full_text, add_special_tokens=False)
            
            print_context(tokens, sys_end, tokenizer, label="System End")
            print_context(tokens, t_stmt_start, tokenizer, label="Target Stmt Start")
            print_context(tokens, t_stmt_end, tokenizer, label="Target Stmt End")
            print_context(tokens, t_start, tokenizer, label="Target Solution Start")
            
            # Show region sizes
            print(f"\n  [Region Token Lengths]")
            print(f"    System         : {sys_end}")
            print(f"    Distractor Pre : {t_stmt_start - sys_end}")
            print(f"    Target Stmt    : {t_stmt_end - t_stmt_start}")
            print(f"    Distractor Post: {t_start - t_stmt_end}")
            print(f"    Target Solution: {len(tokens) - t_start}")
            
        except Exception as e:
            print(f"  Error processing sample {i+1}: {e}")
            
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
