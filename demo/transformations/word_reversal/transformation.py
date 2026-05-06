import re
import random

def apply_word_reversal(text, seed=None):
    if seed:
        random.seed(seed)
        
    # Global word reversal on the entire text.
    # We still use the same tokenizer logic: split strictly by whitespace.
    # This treats punctuation attached to words as part of the word (e.g., "world." is one token).
    
    fragments = re.split(r'(\s+)', text)
    
    parsed_tokens = []
    words_to_reverse = [] 
    
    for frag in fragments:
        if not frag:
            continue
            
        if re.match(r'^\s+$', frag):
            parsed_tokens.append({"type": "space", "content": frag})
        else:
            parsed_tokens.append({"type": "word_slot"})
            words_to_reverse.append(frag)

    # Reverse the collected words from the ENTIRE text
    reversed_words = words_to_reverse[::-1]
    
    # Reconstruct text
    sent_str = ""
    rw_idx = 0
    
    for pt in parsed_tokens:
        if pt["type"] == "space":
            sent_str += pt["content"]
        elif pt["type"] == "word_slot":
            if rw_idx < len(reversed_words):
                sent_str += reversed_words[rw_idx]
                rw_idx += 1
                
    return sent_str

def reverse_word_reversal(text):
    """
    Reverses the word reversal transformation.
    """
    return apply_word_reversal(text)
