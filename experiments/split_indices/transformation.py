
import random
import sys
import os
import nltk

# Ensure NLTK data (punkt and punkt_tab) is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Ensure experiments directory is in path to import context_rot
try:
    from context_rot.generate_systems import generate_systems
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from context_rot.generate_systems import generate_systems

def embed_split_index(text, index_str):
    """
    Splits index_str into 2 parts and embeds them into text 
    as standalone sentences, roughly in the middle and separated.
    """
    # Split index string
    if len(index_str) <= 1:
        part1, part2 = index_str, ""
    else:
        # Constraint: "including at least one letter in each part"
        # index_str format: [[Problem123]]
        # Indices: 01([[) 2(P) ... 8(m) ...
        # Part 1 must include P (index 2) -> split_point >= 3
        # Part 2 must include m (index 8) -> split_point <= 8
        # Strict range [3, 8] ensures splitting strictly inside "Problem".
        split_point = random.randint(3, 8)
            
        part1 = index_str[:split_point]
        part2 = index_str[split_point:]

    # Make them look like sentences (append period if not present?)
    # User example: "[[Probl."
    # We'll validly terminate them with a period to act as sentences.
    # But wait, "[[Pro" isn't a sentence.
    # "Each part will form its own sentence".
    # We append a period.
    part1_sent = part1 + "."
    part2_sent = part2 + "."
    
    parts = [part1_sent, part2_sent]
    random.shuffle(parts) # Allow reverse order
    
    sentences = nltk.sent_tokenize(text)
    
    num_sents = len(sentences)
    
    # Fallback for very short texts
    if num_sents < 3:
        # Insert simply with spaces if we can't do sentence granularity effectively
        # But we try to respect boundaries.
        # Just append/prepend? 
        # Requirement: "middle range".
        # If < 3 sentences, middle is between 0 and 1, or 1 and 2.
        # We'll just insert between 0 and 1, and 1 and 2 if exists.
        
        # If single sentence, forced to break or append/prepend.
        # "not in the first 10% or last 10%".
        # If single sentence is long, we might need whitespace splitting.
        # But for 'generate_systems' content, it's usually multi-sentence.
        # Real problems are usually multi-sentence.
        pass

    # Determine valid range (10% - 90%)
    # Indices represent slots BETWEEN sentences.
    # 0: before s0. num_sents: after last s.
    # Range: [int(N*0.1), int(N*0.9)]
    
    start_idx = max(1, int(num_sents * 0.1))
    end_idx = min(num_sents - 1, int(num_sents * 0.9))
    
    if start_idx >= end_idx:
        # Fallback to full inner range
        start_idx = 1
        end_idx = max(1, num_sents - 1)
        
    valid_indices = list(range(start_idx, end_idx + 1))
    
    if not valid_indices:
        # Just use middle
        valid_indices = [num_sents // 2]
        
    # We need 2 distinct indices.
    if len(valid_indices) == 1:
        idx1 = valid_indices[0]
        # Insert both at same place? Or force another?
        # If only 1 slot, we can put both there: S1. P1. P2. S2.
        # Compatible.
        idx2 = idx1
    else:
        # Pick two indices separated by some distance
        # "far from each other".
        # Try to maximize distance or pick from first and second half of valid range.
        mid_valid = (start_idx + end_idx) // 2
        
        # Ranges: [start, mid] and [mid+1, end]
        range1 = [i for i in valid_indices if i <= mid_valid]
        range2 = [i for i in valid_indices if i > mid_valid]
        
        if not range1: range1 = valid_indices
        if not range2: range2 = valid_indices
        
        idx1 = random.choice(range1)
        idx2 = random.choice(range2)
        
        # Ensure distinct if possible
        if idx1 == idx2 and len(valid_indices) > 1:
             others = [x for x in valid_indices if x != idx1]
             idx2 = random.choice(others)
             
    # Sort indices to insert correctly
    indices = sorted([idx1, idx2])
    
    # Reconstruct
    # If indices are i1, i2.
    # sents[:i1] + [p1] + sents[i1:i2] + [p2] + sents[i2:]
    
    # We shuffled parts earlier, so p1 is parts[0], p2 is parts[1]
    
    final_sents = (
        sentences[:indices[0]] + 
        [parts[0]] + 
        sentences[indices[0]:indices[1]] + 
        [parts[1]] + 
        sentences[indices[1]:]
    )
    
    return " ".join(final_sents)

def apply_split_indices_transformation(problem, seed=None):
    if seed:
        random.seed(seed)
        
    dummy_vars = ["x", "y", "n", "k", "A", "B", "S"]
    distractors = generate_systems(dummy_vars, 30)
    
    all_ids = random.sample(range(100, 999), 31)
    target_id = all_ids[-1]
    distractor_ids = all_ids[:-1]
    
    processed_distractors = []
    for i, dist in enumerate(distractors):
        pid = f"[[Problem{distractor_ids[i]}]]"
        processed = embed_split_index(dist, pid)
        processed_distractors.append(processed)
        
    target_pid = f"[[Problem{target_id}]]"
    processed_real = embed_split_index(problem, target_pid)
    
    block1 = " ".join(processed_distractors[:15])
    block2 = " ".join(processed_distractors[15:])
    
    instruction = f"\n\nYou are to solve Problem{target_id}.\n\n"
    
    final_prompt = block1 + instruction + block2 + "\n\n" + processed_real
    
    return final_prompt

apply_split_indices = apply_split_indices_transformation
