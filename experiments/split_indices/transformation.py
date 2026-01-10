
import random
import sys
import os

# Ensure experiments directory is in path to import context_rot
# Assuming this script is run via evaluate.py which sets up path or run from root
try:
    from context_rot.generate_systems import generate_systems
except ImportError:
    # Fallback if run directly or path issue
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from context_rot.generate_systems import generate_systems

def embed_split_index(text, index_str):
    """
    Splits index_str into 2 parts and embeds them into text 
    at random positions within [10%, 90%] of text length.
    """
    if len(text) < 10: # Safety for very short text
        return text + " " + index_str
        
    # Split index string
    if len(index_str) <= 1:
        part1, part2 = index_str, ""
    else:
        split_point = random.randint(1, len(index_str) - 1)
        part1 = index_str[:split_point]
        part2 = index_str[split_point:]
        
    # Determine positions
    # We want valid range 10%-90%
    start_range = int(len(text) * 0.1)
    end_range = int(len(text) * 0.9)
    
    if start_range >= end_range:
        start_range = 0
        end_range = len(text)
        
    # Insert part1
    pos1 = random.randint(start_range, end_range)
    # Insert with spaces to avoid merging with words
    text_embedded = text[:pos1] + " " + part1 + " " + text[pos1:]
    
    # Insert part2
    # Recalculate range on new text?
    # Or just use similar logic.
    start_range = int(len(text_embedded) * 0.1)
    end_range = int(len(text_embedded) * 0.9)
    if start_range >= end_range:
        start_range = 0
        end_range = len(text_embedded)
        
    pos2 = random.randint(start_range, end_range)
    final_text = text_embedded[:pos2] + " " + part2 + " " + text_embedded[pos2:]
    
    # Reverse order chance?
    # User said "The splits can be placed in reverse order".
    # Since we insert sequentially at random pos, sometimes part2 comes before part1?
    # No, we insert part1 then part2.
    # pos2 refers to index in *modified* text.
    # If pos2 < pos1 (mapped), it's reversed.
    # Random insertion naturally handles random ordering probability.
    
    return final_text

def apply_split_indices_transformation(problem, seed=None):
    if seed:
        random.seed(seed)
        
    # Generate 30 distractors
    dummy_vars = ["x", "y", "n", "k", "A", "B", "S"]
    distractors = generate_systems(dummy_vars, 30)
    
    # IDs
    # 31 unique IDs
    all_ids = random.sample(range(100, 999), 31)
    target_id = all_ids[-1]
    distractor_ids = all_ids[:-1]
    
    # Embed in distractors
    processed_distractors = []
    for i, dist in enumerate(distractors):
        pid = f"[[Problem{distractor_ids[i]}]]"
        processed = embed_split_index(dist, pid)
        processed_distractors.append(processed)
        
    # Embed in Real Problem
    target_pid = f"[[Problem{target_id}]]"
    processed_real = embed_split_index(problem, target_pid)
    
    # Assemble
    # "15 sq_pairs separated by spaces"
    # We'll use space as separator, but since systems explain things, 
    # maybe newlines are better for readability?
    # User said "separated by spaces". We will interpret strictly as " " join 
    # but since systems have newlines, effectively it is space + content.
    
    block1 = " ".join(processed_distractors[:15])
    block2 = " ".join(processed_distractors[15:])
    
    instruction = f"\n\nYou are to solve Problem{target_id}.\n\n"
    
    final_prompt = block1 + instruction + block2 + "\n\n" + processed_real
    
    return final_prompt

apply_split_indices = apply_split_indices_transformation
