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

def get_target_statement_boundary(full_text: str, original: str, target_problem_num: int, tokenizer) -> tuple[int, int]:
    """Finds the token boundaries for the target problem statement in the input prompt.
    The target problem is always the last problem in `original`.
    Returns (stmt_start_token_idx, stmt_end_token_idx) as a half-open [start, end) range."""
    
    # Find the target problem marker in original
    pattern = re.compile(rf"Problem\s*{target_problem_num}\s*:", re.IGNORECASE)
    match = pattern.search(original)
    if not match:
        raise ValueError(f"Could not find 'Problem {target_problem_num}:' in original prompt")
    
    # Find where original appears in full_text
    original_offset = full_text.find(original)
    if original_offset == -1:
        raise ValueError("Could not find original prompt text within full_text")
    
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
    
    print(f"  -> Target statement boundary: tokens [{stmt_start_token}, {stmt_end_token}) = {stmt_end_token - stmt_start_token} tokens")
    
    return stmt_start_token, stmt_end_token

def find_last_complete_sample(filepath):
    """Parse a partial results file to find the last completed sample number.
    Truncates any incomplete sample data at the end of the file.
    Returns the 1-based number of the last complete sample, or 0 if none found."""
    with open(filepath, 'r') as f:
        content = f.read()

    sample_headers = list(re.finditer(r'=== SAMPLE (\d+) DILUTION SUMMARY', content))

    if not sample_headers:
        return 0

    last_header = sample_headers[-1]
    sample_num = int(last_header.group(1))
    remaining = content[last_header.start():]
    separator = '========================================='

    if separator in remaining:
        # Last sample is complete - truncate anything after its separator
        sep_pos = last_header.start() + remaining.index(separator) + len(separator)
        # Include trailing newlines
        while sep_pos < len(content) and content[sep_pos] == '\n':
            sep_pos += 1
        if sep_pos < len(content):
            with open(filepath, 'w') as f:
                f.write(content[:sep_pos])
        return sample_num
    else:
        # Last sample is incomplete - truncate it entirely
        with open(filepath, 'w') as f:
            f.write(content[:last_header.start()])
        if len(sample_headers) >= 2:
            return int(sample_headers[-2].group(1))
        return 0


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
    
    # Check for existing partial results to enable resumption
    resumed = False
    start_idx = 0
    if os.path.exists(out_filepath):
        last_complete = find_last_complete_sample(out_filepath)
        if last_complete > 0:
            start_idx = last_complete
            resumed = True
            print(f"Found existing results file with {last_complete} completed sample(s).")
            print(f"Resuming from sample {start_idx + 1}...")
            if start_idx >= num_samples:
                print("All samples already processed. Nothing to do.")
                return
        else:
            print("Existing file has no complete samples. Starting fresh.")
            with open(out_filepath, 'w') as f:
                f.write(f"DILUTION ANALYSIS FOR: {args.json_file}\n")
                f.write(f"MODEL ID: {model_id}\n")
                f.write(f"DISTRACTORS: {num_distractors}\n")
                f.write("="*80 + "\n")
    else:
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
        if idx < start_idx:
            continue
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
            target_stmt_start_idx, target_stmt_end_idx = get_target_statement_boundary(full_text, original, target_problem_num, tokenizer)
        except ValueError as e:
            print(f"Skipping sample {idx+1} due to boundary error: {e}")
            continue
            
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        total_tokens = len(tokens)
        
        attach_dilution_interceptors(
            model, 
            system_end_idx=system_end_idx,
            target_stmt_start_idx=target_stmt_start_idx,
            target_stmt_end_idx=target_stmt_end_idx,
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
            dpre_scores = layer_scores["distractor_pre"]
            tstmt_scores = layer_scores["target_stmt"]
            dpost_scores = layer_scores["distractor_post"]
            target_scores = layer_scores["target"]
            
            avg_sys = sys_scores.mean(dim=1) * 100.0
            avg_dpre = dpre_scores.mean(dim=1) * 100.0
            avg_tstmt = tstmt_scores.mean(dim=1) * 100.0
            avg_dpost = dpost_scores.mean(dim=1) * 100.0
            avg_tgt = target_scores.mean(dim=1) * 100.0
            
            # Initialize accumulator for this layer if not exists
            if layer_idx not in accumulated_results:
                num_heads = sys_scores.shape[0]
                accumulated_results[layer_idx] = {
                    "system": torch.zeros(num_heads, device='cpu'),
                    "distractor_pre": torch.zeros(num_heads, device='cpu'),
                    "target_stmt": torch.zeros(num_heads, device='cpu'),
                    "distractor_post": torch.zeros(num_heads, device='cpu'),
                    "target": torch.zeros(num_heads, device='cpu')
                }
                
            # Accumulate
            accumulated_results[layer_idx]["system"] += avg_sys.cpu()
            accumulated_results[layer_idx]["distractor_pre"] += avg_dpre.cpu()
            accumulated_results[layer_idx]["target_stmt"] += avg_tstmt.cpu()
            accumulated_results[layer_idx]["distractor_post"] += avg_dpost.cpu()
            accumulated_results[layer_idx]["target"] += avg_tgt.cpu()
            
            # Format individual table
            line1 = (f"\nLayer {layer_idx:2d} Averages - "
                     f"System: {avg_sys.mean().item():.1f}%, "
                     f"DistractorPre: {avg_dpre.mean().item():.1f}%, "
                     f"TargetStmt: {avg_tstmt.mean().item():.1f}%, "
                     f"DistractorPost: {avg_dpost.mean().item():.1f}%, "
                     f"Target: {avg_tgt.mean().item():.1f}%")
            sys_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_sys)]
            dpre_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_dpre)]
            tstmt_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_tstmt)]
            dpost_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_dpost)]
            tgt_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_tgt)]
            
            line2 = f"  System Heads         : " + ", ".join(sys_head_strs)
            line3 = f"  DistractorPre Heads  : " + ", ".join(dpre_head_strs)
            line4 = f"  TargetStmt Heads     : " + ", ".join(tstmt_head_strs)
            line5 = f"  DistractorPost Heads : " + ", ".join(dpost_head_strs)
            line6 = f"  Target Heads         : " + ", ".join(tgt_head_strs)
            
            sample_output_lines.extend([line1, line2, line3, line4, line5, line6])
            
        sample_output_lines.append("=========================================\n")
        sample_text = "\n".join(sample_output_lines)
        
        # Write this sample immediately to the file
        with open(out_filepath, 'a') as f:
            f.write(sample_text + "\n")
            
        successful_samples += 1

    # Generate final aggregated table
    final_lines = []
    if resumed:
        print(f"\nRun was resumed from sample {start_idx + 1}. Aggregation step skipped.")
        with open(out_filepath, 'a') as f:
            f.write(f"\n[Run resumed from sample {start_idx + 1}. Aggregation step skipped.]\n")
    elif successful_samples > 0:
        final_lines.append(f"\n=== AGGREGATED DILUTION SUMMARY (Averaged across {successful_samples} samples) ===")
        for layer_idx in sorted(accumulated_results.keys()):
            avg_sys = accumulated_results[layer_idx]["system"] / successful_samples
            avg_dpre = accumulated_results[layer_idx]["distractor_pre"] / successful_samples
            avg_tstmt = accumulated_results[layer_idx]["target_stmt"] / successful_samples
            avg_dpost = accumulated_results[layer_idx]["distractor_post"] / successful_samples
            avg_tgt = accumulated_results[layer_idx]["target"] / successful_samples
            
            line1 = (f"\nLayer {layer_idx:2d} Averages - "
                     f"System: {avg_sys.mean().item():.1f}%, "
                     f"DistractorPre: {avg_dpre.mean().item():.1f}%, "
                     f"TargetStmt: {avg_tstmt.mean().item():.1f}%, "
                     f"DistractorPost: {avg_dpost.mean().item():.1f}%, "
                     f"Target: {avg_tgt.mean().item():.1f}%")
            sys_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_sys)]
            dpre_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_dpre)]
            tstmt_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_tstmt)]
            dpost_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_dpost)]
            tgt_head_strs = [f"H{h}:{val:.1f}%" for h, val in enumerate(avg_tgt)]
            
            line2 = f"  System Heads         : " + ", ".join(sys_head_strs)
            line3 = f"  DistractorPre Heads  : " + ", ".join(dpre_head_strs)
            line4 = f"  TargetStmt Heads     : " + ", ".join(tstmt_head_strs)
            line5 = f"  DistractorPost Heads : " + ", ".join(dpost_head_strs)
            line6 = f"  Target Heads         : " + ", ".join(tgt_head_strs)
            
            final_lines.extend([line1, line2, line3, line4, line5, line6])
            
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
