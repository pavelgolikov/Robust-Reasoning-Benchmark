import random
# import re

def apply_word_split_swap(text, seed=None):
    if seed:
        random.seed(seed)
        
    # tokens = re.split(r'(\s+)', text)
    tokens = text.split(" ")
    output_tokens = []
    for token in tokens:
        # If whitespace, keep as is
        if not token.strip():
            output_tokens.append(token)
            continue
        
        n = len(token)
        if n < 2:
            output_tokens.append(token)
        else:
            split_idx = n // 2
            part1 = token[:split_idx]
            part2 = token[split_idx:]
            output_tokens.append(part2 + part1)
    return " ".join(output_tokens)

def reverse_word_split_swap(text):
    # Match forward tokenizer: split by whitespace
    tokens = text.split(" ")
    
    output_tokens = []
    for token in tokens:
        if not token.strip(): # whitespace
             output_tokens.append(token)
             continue
             
        n = len(token)
        if n < 2:
            output_tokens.append(token)
        else:
            k = n // 2
            p1 = token[-k:]
            p2 = token[:-k]
            output_tokens.append(p1 + p2)
    return " ".join(output_tokens)
