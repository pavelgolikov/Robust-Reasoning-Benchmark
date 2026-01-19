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

def apply_opposites(text, k=1, seed=None):
    """
    Identifies Verbs and Adjectives.
    Randomly selects 1/k of them (default k=2 means 50%).
    Replaces them with a dynamic antonym from WordNet.
    Fallback: "anti-" prefix.
    """
    if seed is not None:
        random.seed(seed)
        
    doc = nlp(text)
    
    # Group candidates by lowercased word for consistent swapping
    candidates_by_word = {}
    for token in doc:
        if token.pos_ in ["VERB", "ADJ"] and token.is_alpha:
            word_lower = token.text.lower()
            if word_lower not in candidates_by_word:
                candidates_by_word[word_lower] = []
            candidates_by_word[word_lower].append(token)
            
    # Also track ALL alpha tokens by lowercased word to ensure global swapping (Sample 120 Fix)
    all_tokens_by_word = {}
    for token in doc:
        if token.is_alpha:
            wl = token.text.lower()
            if wl not in all_tokens_by_word:
                all_tokens_by_word[wl] = []
            all_tokens_by_word[wl].append(token)
            
    if not candidates_by_word:
        return text
    
    unique_words = list(candidates_by_word.keys())
    num_to_swap = max(1, len(unique_words) // k)
    
    # Randomly select WORDS to swap, not just tokens
    words_to_swap = random.sample(unique_words, num_to_swap)
    
    # Pass 1: Identify "occupied" words (Sample 279 Fix)
    # Use regex splitting to find words even if attached to symbols (e.g. "black" in "black+linewidth")
    # This prevents swapping a word into something that already exists in a code block or formula.
    import re
    tokens_regex = re.split(r'[^a-zA-Z]+', text)
    occupied_words = set(t.lower() for t in tokens_regex if t)
            
    definitions = []
    replacements = {} 
    
    # Pass 2: Select replacements for each chosen WORD
    for word_lower in words_to_swap:
        # Get one of the tokens to check POS (assuming POS is consistent for the word, 
        # or just use the first one's POS as a heuristic).
        # Note: A word might have different POS tags in different contexts (left as VBD vs ADJ).
        # We'll try to find an antonym based on the most common POS or just the first one.
        example_token = candidates_by_word[word_lower][0]
        
        # 1. Try NLTK
        antonym = get_antonym(word_lower, example_token.pos_)
        
        replacement_base = None
        
        if antonym:
            antonym_lower = antonym.lower()
            # CHECK COLLISION: 
            if antonym_lower not in occupied_words:
                 replacement_base = antonym
        
        # 2. Fallback
        if not replacement_base:
            replacement_base = "anti" + word_lower
            if replacement_base in occupied_words:
                 replacement_base = "anti_" + word_lower
                 
        # Apply replacement to ALL instances of this word (Sample 120 Fix)
        # Use keys from all_tokens_by_word if available, otherwise just the candidates
        target_tokens = all_tokens_by_word.get(word_lower, candidates_by_word[word_lower])
        for token in target_tokens:
            word = token.text
            
            # Match case
            if word[0].isupper():
                replacement = replacement_base.capitalize()
            else:
                replacement = replacement_base
                
            replacements[token.i] = replacement
            
            # Add definition
            def_str = f'let "{replacement}" mean "{word}"'
            if def_str not in definitions:
                definitions.append(def_str)
        
        # CLAIM the replacement word globally so no other swap uses it
        occupied_words.add(replacement_base.lower())

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
        
    # 3. Remove the defyn block AND surrounding whitespace artifacts (Sample 279 Fix)
    # The transformation inserts \n\n{block}\n\n. We want to collapse that back to a single space
    # if it was split at a space, or generally clean it up.
    # Note: transformation logic prefers splitting at a space.
    
    # Replace the block and surrounding whitespace with a single space
    # Pattern explanation: 
    # \s* match preceding whitespace (including the \n\n inserted)
    # defyn\{.*?\}\. matches the block
    # \s* match following whitespace
    text_clean = re.sub(r'\s*defyn\{.*?\}\.\s*', ' ', text, flags=re.DOTALL)
    
    # 4. cleanup whitespace
    # Collapse multiple spaces
    text_clean = " ".join(text_clean.split())
    
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
    
    # Create master regex: (?:\b|(?<=\d))(Key1|Key2|...)\b
    # Sample 175 Fix: Use lookbehind (?<=\d) to match words attached to numbers (e.g. 3m)
    
    master_pattern = re.compile(r'(?:\b|(?<=\d))(' + '|'.join(escaped_keys) + r')\b')
    
    def replace_callback(m):
        key = m.group(1)
        # Return mapped value
        return mappings.get(key, key)
        
    reversed_text = master_pattern.sub(replace_callback, text_clean)
    
    return reversed_text

