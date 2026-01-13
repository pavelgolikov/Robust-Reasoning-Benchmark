import spacy
import random

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def _apply_not_not(text, k=2):
    """Inserts 'not not ' before first ADJ and every k-th eligible word (ADJ/NUM)."""
    doc = nlp(text)
    
    eligible_indices = []
    first_adj_index = -1
    
    for i, token in enumerate(doc):
        if token.pos_ in ["ADJ", "NUM"]:
            eligible_indices.append(i)
            if first_adj_index == -1 and token.pos_ == "ADJ":
                first_adj_index = i
                
    insertion_indices = set()
    if first_adj_index != -1:
        insertion_indices.add(first_adj_index)
        
    for count, idx in enumerate(eligible_indices, 1):
        if count % k == 0:
            insertion_indices.add(idx)
            
    output_tokens = []
    for i, token in enumerate(doc):
        if i in insertion_indices:
            output_tokens.append("not not ")
        output_tokens.append(token.text_with_ws)
        
    return "".join(output_tokens)

def _apply_yot(text):
    """
    Inserts 'yot', 'yot yot', or 'not not' before every ADJ/NUM.
    """
    doc = nlp(text)
    
    insertion_indices = []
    for token in doc:
        if token.pos_ in ["ADJ", "NUM"]:
            insertion_indices.append(token.i)
    
    output_tokens = []
    options = ["yot", "yot yot", "not not"]
    
    for i, token in enumerate(doc):
        if i in insertion_indices:
            phrase = random.choice(options)
            output_tokens.append(phrase + " ")
        output_tokens.append(token.text_with_ws)
        
    return "".join(output_tokens)

def apply_not_not_yot(text, k_not=2, seed=None):
    """
    Applies chain: Not Not -> Yot
    No semantic opposites remapping.
    """
    if seed:
        random.seed(seed)
        
    # Phase 1: Not Not
    text_not = _apply_not_not(text, k=k_not)
    
    # Phase 2: Yot
    text_final = _apply_yot(text_not)
    
    return text_final

def reverse_not_not_yot(text):
    """
    Reverses the not-not-yot transformation by removing:
    - 'not not '
    - 'yot '
    - 'yot yot '
    
    Order of operations:
    Remove 'yot' patterns first (as they are unique to this transform).
    Then remove 'not not' patterns.
    """
    import re
    
    # Remove 'yot ' and 'yot yot ' (and any potential multiple yots)
    # Matches 'yot' followed by whitespace, one or more times
    text = re.sub(r'(yot\s+)+', '', text)
    
    # Remove 'not not '
    # Note: Matches 'not not' followed by whitespace.
    text = re.sub(r'not\s+not\s+', '', text)
    
    return text

# Alias for consistent naming schema expected by test scripts
# (If the forward function is named differently in the test script map)
# The test script maps 'not_not_yot' to ... let's check check test_reversibility.py

