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

def apply_opposite_semantic_remapping(text, k=2, seed=None):
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
    
    definitions = []
    replacements = {} 
    
    for token in to_swap:
        word = token.text
        word_lower = word.lower()
        
        # 1. Try NLTK
        antonym = get_antonym(word_lower, token.pos_)
        
        if antonym:
            replacement_base = antonym
        else:
            # 2. Fallback
            replacement_base = "anti" + word_lower
            
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

def _apply_not_not(text, k=3):
    """Inserts 'not not ' before first ADJ and every k-th eligible word (ADJ/NUM)."""
    doc = nlp(text)
    
    eligible_indices = []
    first_adj_index = -1
    
    for i, token in enumerate(doc):
        if token.pos_ in ["ADJ", "NUM"]:
            eligible_indices.append(i)
            if first_adj_index == -1 and token.pos_ == "ADJ":
                first_adj_index = i
                
    insertion_indices = set()
    if first_adj_index != -1:
        insertion_indices.add(first_adj_index)
        
    for count, idx in enumerate(eligible_indices, 1):
        if count % k == 0:
            insertion_indices.add(idx)
            
    output_tokens = []
    for i, token in enumerate(doc):
        if i in insertion_indices:
            output_tokens.append("not not ")
        output_tokens.append(token.text_with_ws)
        
    return "".join(output_tokens)

def _apply_yot(text):
    """
    Inserts 'yot', 'yot yot', or 'not not' before every ADJ/NUM.
    """
    doc = nlp(text)
    
    insertion_indices = []
    for token in doc:
        if token.pos_ in ["ADJ", "NUM"]:
            insertion_indices.append(token.i)
    
    output_tokens = []
    options = ["yot", "yot yot", "not not"]
    
    for i, token in enumerate(doc):
        if i in insertion_indices:
            phrase = random.choice(options)
            output_tokens.append(phrase + " ")
        output_tokens.append(token.text_with_ws)
        
    return "".join(output_tokens)

def apply_opposites_not_yot(text, k_opp=1, k_not=3, seed=None):
    """
    Applies chain: Opposites -> Not Not -> Yot
    """
    if seed:
        random.seed(seed)
        
    # Phase 1: Opposites Replacements
    doc = nlp(text)
    candidates = [t for t in doc if t.pos_ in ["VERB", "ADJ"] and t.is_alpha]
    
    definitions = []
    replacements = {}
    
    if candidates:
        num_to_swap = max(1, len(candidates) // k_opp)
        to_swap = random.sample(candidates, num_to_swap)
        
        for token in to_swap:
            word = token.text
            word_lower = word.lower()
            antonym = get_antonym(word_lower, token.pos_)
            replacement_base = antonym if antonym else "anti" + word_lower
            
            if word[0].isupper():
                replacement = replacement_base.capitalize()
            else:
                replacement = replacement_base
                
            replacements[token.i] = replacement
            
            def_str = f'let "{replacement}" mean "{word}"'
            if def_str not in definitions:
                definitions.append(def_str)
                
    output_tokens = []
    for i, token in enumerate(doc):
        if i in replacements:
            output_tokens.append(replacements[i] + token.whitespace_)
        else:
            output_tokens.append(token.text_with_ws)
    
    text_opp = "".join(output_tokens)
    
    # Phase 2: Not Not
    text_not = _apply_not_not(text_opp, k=k_not)
    
    # Phase 3: Yot
    text_final = _apply_yot(text_not)
    
    if definitions:
        def_block = "defyn{" + ", ".join(definitions) + "}.\n\n"
    else:
        def_block = ""
        
    return def_block + text_final + "\n\n" + def_block
