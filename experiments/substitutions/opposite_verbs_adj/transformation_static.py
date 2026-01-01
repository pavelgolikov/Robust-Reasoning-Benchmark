import spacy
import random
import sys
import os

# Add parent directory to sys.path to find 'antonyms.py'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from antonyms import ANTONYM_DICTIONARY
except ImportError:
    # Fallback if specific file move issue, though sys.path should fix it
    ANTONYM_DICTIONARY = {}

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def apply_static_opposites_transformation(text, k=1, seed=None):
    """
    Applies the LEGACY opposites transformation:
    1. Identifies Verbs/Adjectives.
    2. Checks ANTONYM_DICTIONARY.
    3. Fallback: Prefixes "anti-" to the word.
    """
    if seed is not None:
        random.seed(seed)
        
    doc = nlp(text)
    
    replacements = {}
    definitions = []
    
    # Identify Candidates
    candidates = [] # Tuples of (token, word_lower)
    
    for token in doc:
        if token.pos_ in ["VERB", "ADJ"] and token.is_alpha:
            candidates.append((token, token.text.lower()))

    # Select candidates based on k (fraction)
    # k=1 means 100%
    if not candidates:
        return text

    if k < 1:
        # Interpret k as fraction? Or k items? 
        # In previous logic, k was number of items for 'new word', but expected to be fraction here?
        # Re-reading: The prompt says "k=1 (100% verbs/adj)". So let's handle k=1 as all.
        num_to_select = max(1, int(len(candidates) * k)) if k <= 1 else min(k, len(candidates))
        selected_candidates = random.sample(candidates, num_to_select)
    else:
        # If k >= 1 (int), and we treat k=1 as 100% per user convention for this task?
        # Actually user said "k=1" for 100%. 
        selected_candidates = candidates

    for token, word_lower in selected_candidates:
        original = token.text
        
        # 1. Dictionary Lookup
        if word_lower in ANTONYM_DICTIONARY:
            base_rep = ANTONYM_DICTIONARY[word_lower]
        else:
            # 2. Anti- Fallback
            base_rep = "anti" + word_lower
            
        # Case matching
        if original[0].isupper():
            replacement = base_rep.capitalize()
        else:
            replacement = base_rep
            
        if replacement != original:
            replacements[token.i] = replacement
            
            # Add definition
            def_str = f'let "{replacement}" mean "{original}"'
            if def_str not in definitions:
                definitions.append(def_str)

    # Reconstruct
    output_tokens = []
    for i, token in enumerate(doc):
        if i in replacements:
            output_tokens.append(replacements[i] + token.whitespace_)
        else:
            output_tokens.append(token.text_with_ws)
            
    transformed_text = "".join(output_tokens)
    
    if definitions:
        def_block = "defyn{" + ", ".join(definitions) + "}.\n\n"
    else:
        def_block = ""
        
    return def_block + transformed_text
