from transformers import AutoTokenizer

# Initialize tokenizer (must be a "fast" tokenizer, which is default for most modern models)
tokenizer = AutoTokenizer.from_pretrained("nvidia/OpenReasoning-Nemotron-7B")

def get_token_boundary(full_text: str, boundary_marker: str, tokenizer) -> int:
    # 1. Find exact character index of the boundary
    char_split_idx = full_text.find(boundary_marker)
    if char_split_idx == -1:
        raise ValueError(f"Boundary marker '{boundary_marker}' not found in text.")

    # 2. Tokenize with offset mapping 
    # This returns the (start_char, end_char) position for every token
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]

    # 3. Map character index to token index
    for token_idx, (start_char, end_char) in enumerate(offsets):
        # Find the first token that covers or starts after our character boundary
        if end_char > char_split_idx:
            return token_idx
            
    return len(offsets)

# --- Usage Example ---
full_text = "Here is some distractor math. Problem 3: What is 5+5? Solution: It is 10."
boundary_marker = "Problem 3:"

target_token_idx = get_token_boundary(full_text, boundary_marker, tokenizer)

# Everything before this index is prompt/distractor; everything after is target
print(f"Distractor token count: {target_token_idx}")
print(f"Target token count: {len(tokenizer.encode(full_text)) - target_token_idx}")