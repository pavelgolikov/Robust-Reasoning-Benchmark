import argparse
import json
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attention_interceptor import attach_dilution_interceptors, dilution_results

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
    
    print(f"  -> Heuristic selected group starting with: {longest_group['marker']} (Total Length: {longest_group['total_len']})")
    
    # Tokenize with offset mapping
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    
    # Map character index to token index
    for token_idx, (start_char, end_char) in enumerate(offsets):
        if end_char > char_split_idx:
            return token_idx, char_split_idx
            
    return len(offsets), char_split_idx

def main():
    parser = argparse.ArgumentParser(description="Memory-Efficient Attention Dilution Tracking")
    parser.add_argument("--json_file", type=str, default="/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/compound/results/Qwen_Qwen3-30B-A3B-Thinking-2507/MathArena_aime_2025/Qwen_Qwen3-30B-A3B-Thinking-2507_MathArena_aime_2025_compound_s42_20260330_155901.json", help="Path to compound JSON result file")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size for attention computation")
    args = parser.parse_args()
    
    # Extract model ID from JSON file path (e.g., .../results/Qwen_Qwen3-30B.../dataset/...)
    safe_model_name = os.path.normpath(args.json_file).split(os.sep)[-3]
    model_id = safe_model_name.replace("_", "/", 1)
    
    print(f"Loading data from {args.json_file}")
    print(f"Extracted model ID from path: {model_id}")
    with open(args.json_file, 'r') as f:
        data = json.load(f)
        
    # Find the target entries
    entries = [item for item in data if isinstance(item, dict) and "output" in item]
    num_samples = len(entries)
    print(f"Found {num_samples} valid entries to process.")
    
    if num_samples == 0:
        print("No valid entries found. Exiting.")
        return

    # Extract number of distractors from the first sample
    prompt_problems = re.findall(r"Problem \d+:", entries[0].get("original", ""))
    num_distractors = max(0, len(prompt_problems) - 1)
    
    # Extract datetime from JSON filename
    base_name = os.path.splitext(os.path.basename(args.json_file))[0]
    match = re.search(r'_(\d{8}_\d{6})$', base_name)
    datetime_str = match.group(1) if match else "unknown_time"
    
    out_filepath = f"dilution_{safe_model_name}_{num_distractors}distractors_{datetime_str}.txt"
    print(f"Output will be saved to: {out_filepath}")
    
    # Clear output file at start
    with open(out_filepath, 'w') as f:
        f.write(f"DILUTION ANALYSIS FOR: {args.json_file}\n")
        f.write(f"MODEL ID: {model_id}\n")
        f.write(f"DISTRACTORS: {num_distractors}\n")
        f.write("="*80 + "\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading model...")
    # Conditionally set attention implementation
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    
    # Force SDPA for Qwen/Llama, but let GPT-OSS use its default
    if "gpt-oss" not in model_id.lower() and "gptoss" not in model_id.lower():
        model_kwargs["attn_implementation"] = "sdpa"
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    
    model_type = getattr(model.config, "model_type", "qwen3")
    print(f"Detected model type: {model_type}")
    
    accumulated_results = {}
    successful_samples = 0
    
    for idx, entry in enumerate(entries):
        print(f"\n[{idx+1}/{num_samples}] Processing sample...")
        
        original = entry.get("original", "")
        system_prompt = entry.get("system_prompt", "")
        output = entry.get("output", "")
        
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
        
        try:
            system_end_idx, system_char_idx = get_system_token_boundary(full_text, tokenizer)
            target_start_idx, target_char_idx = get_target_token_boundary(full_text, target_problem_num, tokenizer)
        except ValueError as e:
            print(f"Skipping sample {idx+1} due to boundary error: {e}")
            continue
            
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        total_tokens = len(tokens)
        
        attach_dilution_interceptors(
            model, 
            system_end_idx=system_end_idx,
            target_start_idx=target_start_idx, 
            chunk_size=args.chunk_size,
            model_type=model_type
        )
        
        input_ids = torch.tensor([tokens], device=model.device)
        
        try:
            with torch.no_grad():
                model(input_ids)
        except RuntimeError as e:
            print(f"Skipping sample {idx+1} due to execution error (OOM?): {e}")
            continue
            
        # Extract results for this sample
        sample_output_lines = [f"\n=== SAMPLE {idx+1} DILUTION SUMMARY (Average % mass) ==="]
        
        for layer_idx in sorted(dilution_results.keys()):
            layer_scores = dilution_results[layer_idx]
            sys_scores = layer_scores["system"]
            dist_scores = layer_scores["distractor"]
            target_scores = layer_scores["target"]
            
            avg_sys_per_head = sys_scores.mean(dim=1) * 100.0
            avg_dist_per_head = dist_scores.mean(dim=1) * 100.0
            avg_tgt_per_head = target_scores.mean(dim=1) * 100.0
            
            # Initialize accumulator for this layer if not exists
            if layer_idx not in accumulated_results:
                num_heads = sys_scores.shape[0]
                accumulated_results[layer_idx] = {
                    "system": torch.zeros(num_heads, device='cpu'),
                    "distractor": torch.zeros(num_heads, device='cpu'),
                    "target": torch.zeros(num_heads, device='cpu')
                }
                
            # Accumulate
            accumulated_results[layer_idx]["system"] += avg_sys_per_head.cpu()
            accumulated_results[layer_idx]["distractor"] += avg_dist_per_head.cpu()
            accumulated_results[layer_idx]["target"] += avg_tgt_per_head.cpu()
            
            # Format individual table
            line1 = f"\nLayer {layer_idx:2d} Averages - System: {avg_sys_per_head.mean().item():.1f}%, Distractor: {avg_dist_per_head.mean().item():.1f}%, Target: {avg_tgt_per_head.mean().item():.1f}%"
            sys_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_sys_per_head)]
            dist_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_dist_per_head)]
            tgt_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_tgt_per_head)]
            
            line2 = f"  System Heads     : " + ", ".join(sys_head_strs)
            line3 = f"  Distractor Heads : " + ", ".join(dist_head_strs)
            line4 = f"  Target Heads     : " + ", ".join(tgt_head_strs)
            
            sample_output_lines.extend([line1, line2, line3, line4])
            
        sample_output_lines.append("=========================================\n")
        sample_text = "\n".join(sample_output_lines)
        
        # Write this sample immediately to the file
        with open(out_filepath, 'a') as f:
            f.write(sample_text + "\n")
            
        successful_samples += 1

    # Generate final aggregated table
    final_lines = []
    if successful_samples > 0:
        final_lines.append(f"\n=== AGGREGATED DILUTION SUMMARY (Averaged across {successful_samples} samples) ===")
        for layer_idx in sorted(accumulated_results.keys()):
            avg_sys_per_head = accumulated_results[layer_idx]["system"] / successful_samples
            avg_dist_per_head = accumulated_results[layer_idx]["distractor"] / successful_samples
            avg_tgt_per_head = accumulated_results[layer_idx]["target"] / successful_samples
            
            line1 = f"\nLayer {layer_idx:2d} Averages - System: {avg_sys_per_head.mean().item():.1f}%, Distractor: {avg_dist_per_head.mean().item():.1f}%, Target: {avg_tgt_per_head.mean().item():.1f}%"
            sys_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_sys_per_head)]
            dist_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_dist_per_head)]
            tgt_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_tgt_per_head)]
            
            line2 = f"  System Heads     : " + ", ".join(sys_head_strs)
            line3 = f"  Distractor Heads : " + ", ".join(dist_head_strs)
            line4 = f"  Target Heads     : " + ", ".join(tgt_head_strs)
            
            final_lines.extend([line1, line2, line3, line4])
            
        final_lines.append("========================================================================\n")
        
        full_output = "\n".join(final_lines)
        # print(full_output)
        
        with open(out_filepath, 'a') as f:
            f.write(full_output)
        print(f"Successfully saved aggregated dilution results to the end of {out_filepath}!")
    else:
        print("No successful samples processed. Output file not created.")

if __name__ == "__main__":
    main()
