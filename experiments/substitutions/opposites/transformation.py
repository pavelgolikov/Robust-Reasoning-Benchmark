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
