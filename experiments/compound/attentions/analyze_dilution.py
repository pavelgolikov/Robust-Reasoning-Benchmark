import argparse
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from attention_interceptor import attach_dilution_interceptors, dilution_results

def get_target_token_boundary(full_text: str, target_problem_num: int, tokenizer) -> int:
    """Finds the token boundary for the last occurrence of the target problem marker."""
    # Find all occurrences of the marker
    pattern = re.compile(rf"Problem\s*{target_problem_num}")
    matches = list(pattern.finditer(full_text))
    
    if not matches:
        raise ValueError(f"Boundary marker 'Problem {target_problem_num}' not found in text.")
        
    # Get the last occurrence
    last_match = matches[-1]
    char_split_idx = last_match.start()
    
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
    parser.add_argument("--json_file", type=str, required=True, help="Path to compound JSON result file")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
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
    output = entry.get("output", "")
    full_text = original + output
    
    # Determine the target problem number from the original prompt
    prompt_problems = re.findall(r"Problem \d+:", original)
    num_distractors = max(0, len(prompt_problems) - 1)
    target_problem_num = num_distractors + 1
    
    print(f"Detected {num_distractors} distractors. Target problem is Problem {target_problem_num}.")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    print(f"Finding boundary for 'Problem {target_problem_num}' from the back of the output...")
    target_start_idx = get_target_token_boundary(full_text, target_problem_num, tokenizer)
    
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    total_tokens = len(tokens)
    
    print(f"Total tokens: {total_tokens}")
    print(f"Target start token index: {target_start_idx}")
    print(f"Distractor tokens (I_D): {target_start_idx}")
    print(f"Target tokens (I_T): {total_tokens - target_start_idx}")
    
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
    
    model_type = getattr(model.config, "model_type", "llama")
    print(f"Detected model type: {model_type}")
    
    print(f"Attaching dilution interceptors (chunk_size={args.chunk_size})...")
    attach_dilution_interceptors(
        model, 
        target_start_idx=target_start_idx, 
        chunk_size=args.chunk_size,
        model_type=model_type
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
        "target_start_idx": target_start_idx,
        "total_tokens": total_tokens,
        "token_strings": [tokenizer.decode([t]) for t in tokens[target_start_idx:]] # String representation of target tokens
    }
    
    save_data = {
        "metadata": metadata,
        "dilution_results": dict(dilution_results)
    }
    
    torch.save(save_data, args.output_file)
    print(f"Successfully saved dilution tracking to {args.output_file}!")

if __name__ == "__main__":
    main()
