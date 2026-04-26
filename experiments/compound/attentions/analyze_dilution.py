import argparse
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attention_interceptor import attach_dilution_interceptors, dilution_results

def get_system_token_boundary(full_text: str, tokenizer) -> int:
    """Finds the token boundary for the end of the system prompt (start of first 'Problem N')."""
    pattern = re.compile(r"Problem\s*\d+", re.IGNORECASE)
    match = pattern.search(full_text)
    if not match:
        return 0
        
    char_split_idx = match.start()
    
    # Tokenize with offset mapping
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    
    # Map character index to token index
    for token_idx, (start_char, end_char) in enumerate(offsets):
        if end_char > char_split_idx:
            return token_idx
            
    return len(offsets)

def get_target_token_boundary(full_text: str, target_problem_num: int, tokenizer) -> int:
    """Finds the token boundary for the longest contiguous problem-solving block."""
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
    
    print(f"  -> Heuristic selected group starting with: {longest_group['marker']} (Total Length: {longest_group['total_len']})")
    
    # Tokenize with offset mapping
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    
    # Map character index to token index
    for token_idx, (start_char, end_char) in enumerate(offsets):
        if end_char > char_split_idx:
            return token_idx
            
    return len(offsets)

def main():
    parser = argparse.ArgumentParser(description="Memory-Efficient Attention Dilution Tracking")
    parser.add_argument("--json_file", type=str, default="/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/compound/results/Qwen_Qwen3-30B-A3B-Thinking-2507/MathArena_aime_2025/Qwen_Qwen3-30B-A3B-Thinking-2507_MathArena_aime_2025_compound_s42_20260330_155901.json", help="Path to compound JSON result file")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507", help="HuggingFace model ID")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample in the JSON file")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for attention computation")
    parser.add_argument("--output_file", type=str, default="dilution_results.pt", help="Path to save output .pt file")
    args = parser.parse_args()
    
    print(f"Loading data from {args.json_file} (index {args.index})")
    with open(args.json_file, 'r') as f:
        data = json.load(f)
        
    # Find the target entry
    entries = [item for item in data if isinstance(item, dict) and "output" in item]
    if args.index >= len(entries):
        raise IndexError(f"Index {args.index} out of bounds. Found {len(entries)} valid entries.")
        
    entry = entries[args.index]
    original = entry.get("original", "")
    system_prompt = entry.get("system_prompt", "")
    output = entry.get("output", "")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Reconstruct the exact prompt given to the model using the chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": original}
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = formatted_prompt + output
    
    # Determine the target problem number from the original prompt
    prompt_problems = re.findall(r"Problem \d+:", original)
    num_distractors = max(0, len(prompt_problems) - 1)
    target_problem_num = num_distractors + 1
    
    print(f"Detected {num_distractors} distractors. Target problem is Problem {target_problem_num}.")
    
    print("Finding system prompt boundary...")
    system_end_idx = get_system_token_boundary(full_text, tokenizer)
    
    print(f"Finding boundary for 'Problem {target_problem_num}' using the longest contiguous block heuristic...")
    target_start_idx = get_target_token_boundary(full_text, target_problem_num, tokenizer)
    
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    total_tokens = len(tokens)
    
    print(f"Total tokens: {total_tokens}")
    print(f"System tokens: {system_end_idx}")
    print(f"Distractor tokens: {target_start_idx - system_end_idx}")
    print(f"Target tokens: {total_tokens - target_start_idx}")
    
    system_text = tokenizer.decode(tokens[:system_end_idx])
    print("\n--- SYSTEM REGION ---")
    print(system_text)
    print("---------------------\n")
    
    # Decode to show boundary for sanity check
    distractor_text = tokenizer.decode(tokens[:target_start_idx])
    target_text = tokenizer.decode(tokens[target_start_idx:])
    
    print("\n--- BOUNDARY CHECK ---")
    print(f"Last 100 chars of distractor phase: {repr(distractor_text[-100:])}")
    print(f"First 100 chars of target phase: {repr(target_text[:100])}")
    print("----------------------\n")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa" # Standard scaled dot product attention
    )
    
    # model_type = getattr(model.config, "model_type", "llama")
    # print(f"Detected model type: {model_type}")
    
    print(f"Attaching dilution interceptors (chunk_size={args.chunk_size})...")
    attach_dilution_interceptors(
        model, 
        system_end_idx=system_end_idx,
        target_start_idx=target_start_idx, 
        chunk_size=args.chunk_size
    )
    
    input_ids = torch.tensor([tokens], device=model.device)
    
    print("Executing forward pass (this may take a moment)...")
    with torch.no_grad():
        model(input_ids)
        
    print("Forward pass complete. Gathering results...")
    
    # Format metadata
    metadata = {
        "model_id": args.model_id,
        "json_file": args.json_file,
        "sample_index": args.index,
        "chunk_size": args.chunk_size,
        "target_problem_num": target_problem_num,
        "system_end_idx": system_end_idx,
        "target_start_idx": target_start_idx,
        "total_tokens": total_tokens,
        "token_strings": [tokenizer.decode([t]) for t in tokens[target_start_idx:]] # String representation of target tokens
    }
    
    save_data = {
        "metadata": metadata,
        "dilution_results": dict(dilution_results)
    }
    
    print("\n=== DILUTION SUMMARY (Average % mass) ===")
    for layer_idx in sorted(dilution_results.keys()):
        layer_scores = dilution_results[layer_idx]
        sys_scores = layer_scores["system"]
        dist_scores = layer_scores["distractor"]
        target_scores = layer_scores["target"]
        
        avg_sys_per_head = sys_scores.mean(dim=1) * 100.0
        avg_dist_per_head = dist_scores.mean(dim=1) * 100.0
        avg_tgt_per_head = target_scores.mean(dim=1) * 100.0
        
        print(f"\nLayer {layer_idx:2d} Averages - System: {avg_sys_per_head.mean().item():.1f}%, Distractor: {avg_dist_per_head.mean().item():.1f}%, Target: {avg_tgt_per_head.mean().item():.1f}%")
        
        # sys_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_sys_per_head)]
        # dist_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_dist_per_head)]
        # tgt_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_tgt_per_head)]
        
        # print(f"  System Heads     : " + ", ".join(sys_head_strs))
        # print(f"  Distractor Heads : " + ", ".join(dist_head_strs))
        # print(f"  Target Heads     : " + ", ".join(tgt_head_strs))
    print("=========================================\n")
    
    torch.save(save_data, args.output_file)
    print(f"Successfully saved full raw dilution tracking tensors to {args.output_file}!")

if __name__ == "__main__":
    main()
