
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

# Ensure experiments directory is in path to import
try:
    from context_saturation.generate_systems import generate_systems
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from context_saturation.generate_systems import generate_systems

def embed_split_index(text, index_str):
    """
    Splits index_str into 2 parts and embeds them into text 
    as standalone sentences, roughly in the middle and separated.
    """
    # Split index string
    if len(index_str) <= 1:
        part1, part2 = index_str, ""
    else:
        split_point = random.randint(3, 8)
            
        part1 = index_str[:split_point]
        part2 = index_str[split_point:]

    part1_sent = part1 + "."
    part2_sent = part2 + "."
    
    parts = [part1_sent, part2_sent]
    random.shuffle(parts) # Allow reverse order
    
    sentences = nltk.sent_tokenize(text)
    
    num_sents = len(sentences)
    
    # Fallback for very short texts
    if num_sents < 3:
        pass

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
        idx2 = idx1
    else:
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
    
    final_sents = (
        sentences[:indices[0]] + 
        [parts[0]] + 
        sentences[indices[0]:indices[1]] + 
        [parts[1]] + 
        sentences[indices[1]:]
    )
    
    return " ".join(final_sents)

def apply_context_saturation(problem, num_distractors, seed=None, problem_variables=None):
    if seed:
        random.seed(seed)
        
    dummy_variables = ["x", "y", "n", "k", "A", "B", "S"]
    if problem_variables and len(problem_variables) > 0:
        variables_to_use = problem_variables
        if len(problem_variables) < 2:
            variables_to_use += dummy_variables
    else:
        variables_to_use = dummy_variables
        
    distractors = generate_systems(variables_to_use, num_distractors)
    
    all_ids = random.sample(range(100, 999), num_distractors + 1)
    target_id = all_ids[-1]
    distractor_ids = all_ids[:-1]
    
    processed_distractors = []
    for i, dist in enumerate(distractors):
        pid = f"[[Problem{distractor_ids[i]}]]"
        processed = embed_split_index(dist, pid)
        processed_distractors.append(processed)
        
    target_pid = f"[[Problem{target_id}]]"
    processed_real = embed_split_index(problem, target_pid)
    
    block1 = " ".join(processed_distractors[:(num_distractors // 2)])
    block2 = " ".join(processed_distractors[(num_distractors // 2):])
    
    instruction = f"\n\nYou are to solve Problem{target_id} using standard mathematical operations.\n\n"
    
    final_prompt = block1 + instruction + block2 + "\n\n" + processed_real
    
    return final_prompt

def reverse_context_saturation(text):
    """
    Reverses context saturation by:
    1. Extracting the last paragraph (the real problem).
    2. Removing the embedded split index sentences (e.g., '[[Prob.' and 'lem123]].').
    """
    import re
    
    # 1. Extract last paragraph
    # The transformation adds "\n\n" + processed_real at the end.
    # However, inputs might contain \n\n naturally.
    # But the saturation structure is [Distractors] \n\n [Processed Real].
    # Distractors also don't contain \n\n (they are spaces-joined).
    # So splitting by \n\n and taking the last part is safe.
    
    parts = text.split("\n\n")
    real_problem_text = parts[-1]
    
    # 2. Remove embedded split ID parts
    # Part 1: Starts with "[[P" (from "[[Problem..."), ends with dot.
    # Since split index is random(3, 8), and "[[Problem" is 9 chars,
    # Part 1 always contains at least "[[P".
    # Regex: `\[\[P.*?\.\s*` matching "[[Prob." 
    real_problem_text = re.sub(r'\[\[P.*?\.\s*', '', real_problem_text)
    
    # Part 2: Ends with "]]."
    # It is the suffix of "[[ProblemID]]".
    # It contains no spaces.
    # Regex: `\S*\]\]\.\s*` 
    real_problem_text = re.sub(r'\S*\]\]\.\s*', '', real_problem_text)
    
    return real_problem_text.strip()


