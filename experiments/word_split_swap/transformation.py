import spacy
import random

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def transform_word(word):
    n = len(word)
    if n < 2:
        return word
        
    # Calculate split point
    # If even: n/2
    # If odd: (n-1)/2 (since 1st part has 1 less char than 2nd: L1 + (L1+1) = n => 2L1=n-1)
    split_idx = n // 2
    
    part1 = word[:split_idx]
    part2 = word[split_idx:]
    
    # Swap
    return part2 + part1

def apply_word_split_swap(text, seed=None):
    if seed:
        random.seed(seed)
        
    # User requested literal split by space
    # We use split(' ') which consumes single spaces, or split() which consumes all whitespace?
    # "literally split by space" might mean text.split(' ').
    # But usually we want to preserve newlines/mulit-spaces.
    # Safe approach: re.split(r'(\s+)', text) to keep delimiters (whitespace).
    
    import re
    tokens = re.split(r'(\s+)', text)
    
    output_tokens = []
    for token in tokens:
        # If whitespace, keep as is
        if not token.strip():
            output_tokens.append(token)
            continue
            
        # Transform token
        # Formula:
        # n = len
        # even: split n/2
        # odd: split (n-1)/2. (1st part length = (n-1)/2)
        # Note: (n-1)//2 is the same as n//2 in integer arithmetic for odd n.
        # e.g. 5 // 2 = 2. (5-1)/2 = 2.
        # e.g. 4 // 2 = 2.
        # So split_idx = n // 2 works for both cases as per description?
        # "If odd... first part has one character less than second part"
        # n=5. L1 + L2 = 5. L1 = L2 - 1. 2*L2 - 1 = 5 => 2*L2=6 => L2=3. L1=2.
        # First part length 2.
        # n // 2 = 2.
        # So yes, n // 2 is the correct split index for both.
        
        n = len(token)
        if n < 2:
            output_tokens.append(token)
        else:
            split_idx = n // 2
            part1 = token[:split_idx]
            part2 = token[split_idx:]
            output_tokens.append(part2 + part1)
            
    return "".join(output_tokens)

def reverse_word_split_swap(text):
    import re
    # Match forward tokenizer: split by whitespace
    tokens = re.split(r'(\s+)', text)
    
    output_tokens = []
    for token in tokens:
        if not token.strip(): # whitespace
             output_tokens.append(token)
             continue
             
        n = len(token)
        if n < 2:
            output_tokens.append(token)
        else:
            # Forward: p1 = token[:k], p2 = token[k:]. out = p2 + p1.
            # k = n // 2.
            # p1 part (length k) is now at the end.
            # p2 part (length n-k) is now at the start.
            
            k = n // 2
            p1 = token[-k:]
            p2 = token[:-k]
            
            output_tokens.append(p1 + p2)
            
    return "".join(output_tokens)
