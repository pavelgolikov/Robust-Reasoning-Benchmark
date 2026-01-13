import spacy
import random
import nltk
from nltk.corpus import wordnet

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def get_antonym(word, pos_tag):
    """
    Finds an antonym for a word using NLTK WordNet.
    Returns None if no lexical antonym is found.
    """
    antonyms = []
    
    # Map Spacy POS to WordNet POS
    wn_pos = None
    if pos_tag == "ADJ":
        wn_pos = wordnet.ADJ
    elif pos_tag == "VERB":
        wn_pos = wordnet.VERB
    # elif pos_tag == "ADV":
    #     wn_pos = wordnet.ADV
    # elif pos_tag == "NOUN":
    #     wn_pos = wordnet.NOUN
        
    for syn in wordnet.synsets(word, pos=wn_pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.append(antonym.name())
                
    if antonyms:
        # Sort by frequency/length or just random?
        # Use set to uniques
        unique_antonyms = list(set(antonyms))
        # Prioritize single words over multi-word phrases if any
        unique_antonyms.sort(key=lambda x: (len(x.split('_')), len(x)))
        return unique_antonyms[0].replace('_', ' ')
        
    return None

def apply_opposites(text, k=2, seed=None):
    """
    Identifies Verbs and Adjectives.
    Randomly selects 1/k of them (default k=2 means 50%).
    Replaces them with a dynamic antonym from WordNet.
    Fallback: "anti-" prefix.
    """
    if seed is not None:
        random.seed(seed)
        
    doc = nlp(text)
    
    candidates = []
    for token in doc:
        if token.pos_ in ["VERB", "ADJ"] and token.is_alpha:
            candidates.append(token)
            
    if not candidates:
        return text
        
    num_to_swap = max(1, len(candidates) // k)
    to_swap = random.sample(candidates, num_to_swap)
    to_swap_indices = set(t.i for t in to_swap)

    # 2-Pass Strategy:
    # Pass 1: Identify "occupied" words - words that will remain in the text.
    # This enables swaps (e.g. "good bad" -> "bad good") because "good" and "bad" are NOT in occupied
    # during selection, allowing them to be used as targets.
    
    occupied_words = set()
    for token in doc:
        if token.i not in to_swap_indices:
            occupied_words.add(token.text.lower())
            
    definitions = []
    replacements = {} 
    
    # Pass 2: Select replacements
    # Sort to ensure deterministic order if seed is set? random.sample handling order nicely?
    # We'll just iterate to_swap (which is a list from random.sample)
    
    for token in to_swap:
        word = token.text
        word_lower = word.lower()
        
        # 1. Try NLTK
        antonym = get_antonym(word_lower, token.pos_)
        
        replacement_base = None
        
        if antonym:
            antonym_lower = antonym.lower()
            # CHECK COLLISION: 
            # Can we use this antonym? 
            # Only if it's not currently occupying the text (unswapped) 
            # AND not already claimed by another replacement in this loop.
            if antonym_lower not in occupied_words:
                replacement_base = antonym
        
        # 2. Fallback
        if not replacement_base:
            replacement_base = "anti" + word_lower
            # Safety: Ensure fallback is unique too
            if replacement_base in occupied_words:
                 replacement_base = "anti_" + word_lower
            
        # Match case
        if word[0].isupper():
            replacement = replacement_base.capitalize()
        else:
            replacement = replacement_base
            
        replacements[token.i] = replacement
        
        # CLAIM the word so no one else can use it
        occupied_words.add(replacement.lower())
        
        # Add definition
        def_str = f'let "{replacement}" mean "{word}"'
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
        def_block = "defyn{" + ", ".join(definitions) + "}."
        
        # Insert in the middle
        mid = len(transformed_text) // 2
        # Find nearest space
        left_mid = transformed_text.rfind(' ', 0, mid)
        right_mid = transformed_text.find(' ', mid)
        
        if left_mid == -1: split_idx = right_mid
        elif right_mid == -1: split_idx = left_mid
        else:
            if (mid - left_mid) < (right_mid - mid): split_idx = left_mid
            else: split_idx = right_mid
            
        if split_idx == -1: split_idx = mid
        
        top_half = transformed_text[:split_idx]
        bottom_half = transformed_text[split_idx:]
        
        final_text = top_half + "\n\n" + def_block + "\n\n" + bottom_half
    else:
        final_text = transformed_text
        
    return final_text

def reverse_opposites(text):
    """
    Reverses the opposites transformation by parsing the defyn block 
    and applying the definitions to revert antonyms.
    Uses word-boundary regex to avoid substring collisions and handles swaps simultaneously.
    """
    import re
    
    # 1. Find and Extract defyn block
    pattern = re.compile(r'defyn\{(.*?)\}\.', re.DOTALL)
    match = pattern.search(text)
    
    if not match:
        return text
    
    def_content = match.group(1)
    
    # 2. Parse definitions
    # let "replacement" mean "word"
    # Note: replacements might be multi-word? 
    # The transformation function standardizes antonyms (replacing '_' with ' ').
    # So "replacement" could be "fall apart".
    # But usually it's single word or "anti" + word.
    
    def_pattern = re.compile(r'let "(.*?)" mean "(.*?)"')
    mappings = {}
    
    for m in def_pattern.finditer(def_content):
        replacement = m.group(1)
        original = m.group(2)
        mappings[replacement] = original
        
    # 3. Remove the defyn block
    text_clean = text.replace(match.group(0), "")
    
    # 4. cleanup whitespace
    text_clean = re.sub(r'\n{3,}', '\n\n', text_clean).strip()
    
    # 5. Apply replacements
    # To handle swaps (A->B, B->A) and avoid substring issues, use regex with \b
    # Sort keys by length descending to prioritize longer matches (if overlaps exist)
    
    if not mappings:
        return text_clean
        
    # Escape keys for regex
    escaped_keys = [re.escape(k) for k in sorted(mappings.keys(), key=len, reverse=True)]
    
    # Create master regex: \b(Key1|Key2|...)\b
    # Note: If antonym contains spaces, \b works at ends.
    # "fall apart" -> \bfall apart\b matches "fall apart".
    
    master_pattern = re.compile(r'\b(' + '|'.join(escaped_keys) + r')\b')
    
    def replace_callback(m):
        key = m.group(1)
        # Return mapped value
        return mappings.get(key, key)
        
    reversed_text = master_pattern.sub(replace_callback, text_clean)
    
    return reversed_text

