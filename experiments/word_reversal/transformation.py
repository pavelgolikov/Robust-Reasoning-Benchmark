import re
import random

def apply_word_reversal(text, seed=None):
    if seed:
        random.seed(seed)
        
    # User requested simple splitting on periods.
    parts = text.split('.')
    output_pieces = []
    
    for part in parts:
        # Tokenizer Logic:
        # Split strictly by whitespace.
        # This treats "$9$-kilometer-long" as one word, preventing merging issues.
        
        fragments = re.split(r'(\s+)', part)
        
        parsed_tokens = []
        words_to_reverse = [] 
        
        for frag in fragments:
            if not frag:
                continue
                
            if re.match(r'^\s+$', frag):
                parsed_tokens.append({"type": "space", "content": frag})
            else:
                # Pure Space-Based Tokenization:
                # Everything that is not space is part of the word.
                # Punctuation stays attached.
                parsed_tokens.append({"type": "word_slot"})
                words_to_reverse.append(frag)

        
        # Reverse the collected words
        reversed_words = words_to_reverse[::-1]
        
        # Reconstruct sentence fragment
        sent_str = ""
        rw_idx = 0
        
        for pt in parsed_tokens:
            if pt["type"] == "space" or pt["type"] == "fixed":
                sent_str += pt["content"]
            elif pt["type"] == "word_slot":
                if rw_idx < len(reversed_words):
                    sent_str += reversed_words[rw_idx]
                    rw_idx += 1
                # No suffix punct to add

                    
        output_pieces.append(sent_str)
        
    # Rejoin with periods
    return ".".join(output_pieces)

def reverse_word_reversal(text):
    """
    Reverses the word reversal transformation.
    """
    return apply_word_reversal(text)
