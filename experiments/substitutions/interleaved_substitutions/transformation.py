
import spacy
import random
import itertools
import math
from collections import defaultdict

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def chunk_string(text, chunk_size):
    # Flatten newlines to spaces
    flat = text.replace('\n', ' ').strip()
    return [flat[i:i+chunk_size] for i in range(0, len(flat), chunk_size)]

def get_pos_words(doc):
    """
    Extracts words by POS tag.
    Returns dict: POS -> list of unique words
    """
    pos_map = defaultdict(set)
    # Define interesting POS tags
    target_tags = ["VERB", "NOUN", "ADJ", "ADV"]
    
    for token in doc:
        if token.pos_ in target_tags and token.is_alpha and not token.is_stop:
            pos_map[token.pos_].add(token.text)
            
    # Convert sets to lists
    return {k: list(v) for k, v in pos_map.items()}

def apply_substitution_logic(doc_a, doc_b, k=1, seed=None):
    """
    Replaces words in doc_a with words from doc_b.
    k=1 implies replacing all candidates if substitutes are available. (As 1/k = 100%)
    """
    if seed:
        random.seed(seed)

    words_a = get_pos_words(doc_a)
    words_b = get_pos_words(doc_b)
    
    definitions = []
    replacements = {} # index -> replacement
    
    target_tags = ["VERB", "NOUN", "ADJ", "ADV"]
    
    for pos in target_tags:
        candidates_a = words_a.get(pos, [])
        candidates_b = words_b.get(pos, [])
        
        # Remove common words from lists to avoid identity mapping confusion or trivial swaps?
        # User said: "remove common words from each list"
        set_a = set(candidates_a)
        set_b = set(candidates_b)
        common = set_a.intersection(set_b)
        
        clean_a = [w for w in candidates_a if w not in common]
        clean_b = [w for w in candidates_b if w not in common]
        
        # Shuffle for randomness
        random.shuffle(clean_a)
        random.shuffle(clean_b)
        
        # Determine how many to swap
        # "replace top k words" with default k=1 (meaning all/most?)
        # Previous logic used k as divisor (1/k). If k=1, we take all.
        num_candidates = len(clean_a)
        num_replacements = len(clean_b)
        
        # We can only swap as many as we have unique replacements for
        limit = min(num_candidates, num_replacements)
        
        # Map words
        mapping = {}
        for i in range(limit):
             target_word = clean_a[i]
             replacement_word = clean_b[i]
             mapping[target_word] = replacement_word
             
             # Create definition
             # "Let 'replacement' mean 'target'"
             def_str = f'let "{replacement_word}" mean "{target_word}"'
             definitions.append(def_str)

        # Record replacements by token index
        if mapping:
            for token in doc_a:
                if token.text in mapping:
                    # Check POS to be safe, though we filtered by POS earlier on word extraction
                    # A word might have multiple POS, but we want to replace the instance that matches
                    if token.pos_ == pos:
                        replacements[token.i] = mapping[token.text]

    # Reconstruct Text A
    output_tokens = []
    for i, token in enumerate(doc_a):
        if i in replacements:
            output_tokens.append(replacements[i] + token.whitespace_)
        else:
            output_tokens.append(token.text_with_ws)
    
    transformed_a_text = "".join(output_tokens)
    
    def_block = ""
    if definitions:
        def_block = "defyn{" + ", ".join(definitions) + "}.\n"
        
    return transformed_a_text, def_block

def apply_interleaved_substitutions(problem_a, problem_b, k=1, seed=None):
    """
    1. Transforms A using replacements from B.
    2. Interleaves Transformed A and Original B (chunks of 60 chars).
    3. Adds tags <Problem A> / <Problem B>.
    4. Splits result in half and inserts defyn block.
    """
    doc_a = nlp(problem_a)
    doc_b = nlp(problem_b)
    
    # 1. Substitute
    transformed_a, def_block = apply_substitution_logic(doc_a, doc_b, k, seed)
    
    # 2. Interleave
    chunk_size = 60
    chunks_a = chunk_string(transformed_a, chunk_size)
    chunks_b = chunk_string(problem_b, chunk_size)
    
    # Ensure not empty
    if not chunks_a: chunks_a = [""]
    if not chunks_b: chunks_b = [""]
    
    # Tag chunks
    tagged_a = [c + " <Problem A>" for c in chunks_a]
    tagged_b = [c + " <Problem B>" for c in chunks_b]
    
    max_len = max(len(tagged_a), len(tagged_b))
    
    # Cycle shorter
    input_a = list(itertools.islice(itertools.cycle(tagged_a), max_len))
    input_b = list(itertools.islice(itertools.cycle(tagged_b), max_len))
    
    interleaved_lines = []
    for a, b in zip(input_a, input_b):
        interleaved_lines.append(a)
        interleaved_lines.append(b)
        
    # 4. Insert Defyn Block in Middle
    # "split it horizontally roughly in half, give ample space"
    total_lines = len(interleaved_lines)
    mid_point = total_lines // 2
    
    top_half = interleaved_lines[:mid_point]
    bottom_half = interleaved_lines[mid_point:]
    
    # Spacing
    separator = "\n\n"
    
    final_text = "\n".join(top_half) + separator + def_block + separator + "\n".join(bottom_half)
    
    return final_text
    
