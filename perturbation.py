import spacy

# Load the spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def apply_not_not_perturbation(text, k=3):
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
            # print(f"DEBUG: Found eligible word '{token.text}' ({token.pos_}) at index {i}")
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
    # "Every k-th eligible word" -> 1-based index in the eligible list? 
    # e.g. if eligible are at [2, 5, 8, 10], and k=3. 3rd eligible is at 8.
    for count, idx in enumerate(eligible_indices, 1):
        if count % k == 0:
            insertion_indices.add(idx)
            
    # Reconstruct the text
    # We will build a list of strings, adding "not not " where appropriate
    result = []
    for i, token in enumerate(doc):
        if i in insertion_indices:
            result.append("not not ")
            
        # Append the token text (preserving whitespace if possible, though spacy token.text + token.whitespace_ is standard)
        result.append(token.text_with_ws)
        
    return "".join(result)

import random

def apply_yot_perturbation(text, k=3):
    """
    Inserts 'yot', 'yot yot', or 'not not' before the first adjective and every k-th eligible word thereafter.
    Eligible words are adjectives (ADJ) and numbers (NUM).
    
    Args:
        text (str): The input text (problem statement).
        k (int): Interval for insertion on eligible words.
        
    Returns:
        str: The perturbed text.
    """
    doc = nlp(text)
    
    eligible_indices = []
    adj_indices = []
    
    for i, token in enumerate(doc):
        if token.pos_ == "ADJ":
            adj_indices.append(i)
        elif token.pos_ == "NUM":
            eligible_indices.append(i)
                
    insertion_indices = set()
    
    # User Request: Put something in front of EVERY adjective
    for idx in adj_indices:
        insertion_indices.add(idx)
        
    # Also keep the k-th logic for numbers (or other eligible words if we expand)
    for count, idx in enumerate(eligible_indices, 1):
        if count % k == 0:
            insertion_indices.add(idx)
            
    result = []
    options = ["yot ", "yot yot ", "not not "]
    
    for i, token in enumerate(doc):
        if i in insertion_indices:
            # Randomly choose one of the options
            result.append(random.choice(options))
            
        result.append(token.text_with_ws)
        
    return "".join(result)

def main():
    # Only for quick testing if run directly
    text = "The quick brown fox jumps over the lazy dog."
    print(f"Original: {text}")
    print(f"Not Not: {apply_not_not_perturbation(text, k=2)}")
    print(f"New Word: {apply_new_word_perturbation(text, k=2)}")

if __name__ == "__main__":
    main()
