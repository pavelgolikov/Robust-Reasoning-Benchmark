import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def apply_not_not_transformation(text, k=3):
    """
    Inserts 'not not ' before the first adjective and every k-th eligible word thereafter.
    Eligible words are adjectives (ADJ) and numbers (NUM).
    
    Args:
        text (str): The input text (problem statement).
        k (int): Interval for insertion on eligible words.
        
    Returns:
        str: The perturbed text.
    """
    doc = nlp(text)
    
    eligible_indices = []
    first_adj_index = -1
    
    # Identify eligible words and the first adjective
    for i, token in enumerate(doc):
        # Check if token is adjective or number
        if token.pos_ in ["ADJ", "NUM"]:
            eligible_indices.append(i)
            # Record first adjective index
            if first_adj_index == -1 and token.pos_ == "ADJ":
                first_adj_index = i
                
    # Determine which indices should have "not not " inserted
    insertion_indices = set()
    
    # Rule 1: Always insert in front of the first adjective
    if first_adj_index != -1:
        insertion_indices.add(first_adj_index)
        
    # Rule 2: Insert in front of every k-th eligible word
    for count, idx in enumerate(eligible_indices, 1):
        if count % k == 0:
            insertion_indices.add(idx)
            
    # Reconstruct the text
    output_tokens = []
    for i, token in enumerate(doc):
        if i in insertion_indices:
            output_tokens.append("not not ")
        output_tokens.append(token.text_with_ws)
        
    return "".join(output_tokens)
