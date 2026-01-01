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
        
    if wn_pos:
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.append(antonym.name())
                
    if antonyms:
        unique_antonyms = list(set(antonyms))
        unique_antonyms.sort(key=lambda x: (len(x.split('_')), len(x)))
        return unique_antonyms[0].replace('_', ' ')
        
    return None

def generate_random_number(original_text):
    """
    Generates a random number of the same type and sign.
    Range is pseudo-randomized between 1 and 1000 for variety, 
    preserving sign.
    """
    try:
        if '.' in original_text:
            val = float(original_text)
            is_float = True
        else:
            val = int(original_text)
            is_float = False
    except ValueError:
        return None  # Not a number

    if val == 0:
        return None # Skip zero

    sign = 1 if val > 0 else -1
    
    # Generate random magnitude
    new_mag = random.randint(1, 1000)
    
    if is_float:
        # Add some decimals
        new_mag += random.random()
        return f"{sign * new_mag:.2f}"
    else:
        return str(sign * new_mag)


def apply_opposite_and_number_transformation(text, k=1, seed=None):
    """
    Applies TWO transformations:
    1. Replaces 100% (or k fraction, but defaulting to 100% per user request) 
       of Verbs and Adjectives with Antonyms.
    2. Replaces ALL numbers (except 0) with valid random numbers.
    """
    if seed is not None:
        random.seed(seed)
        
    doc = nlp(text)
    
    replacements = {}
    definitions = []
    
    # Identify Candidates
    candidates = [] # Tuples of (token, type)
    
    for token in doc:
        # Word Antonyms
        if token.pos_ in ["VERB", "ADJ"] and token.is_alpha:
            candidates.append((token, "word"))
        # Numbers
        elif token.pos_ == "NUM" or token.like_num:
            # Simple check if it's actually a digit string or simple number word
            # For simplicity, we target digit strings mainly, unless Spacy parses words like 'ten' as NUM
            # User said "integer or float", implying digits.
            # Let's check if it parses as a number
            try:
                float(token.text)
                candidates.append((token, "number"))
            except ValueError:
                pass

    # Processing
    # For Words: Use k strategy (default k=1 means all)
    # For Numbers: Replace ALL (equivalent to k=1)
    
    for token, type_ in candidates:
        original = token.text
        replacement = None
        
        if type_ == "word":
            # Just do 100% for this experiment per user request context ("implement on top")
            # But let's respect k if passed (though likely 1 here)
            # Actually, user said 100% for opposites.
            word_lower = original.lower()
            antonym = get_antonym(word_lower, token.pos_)
            
            if antonym:
                base_rep = antonym
            else:
                base_rep = "anti" + word_lower
            
            # Case matching
            if original[0].isupper():
                replacement = base_rep.capitalize()
            else:
                replacement = base_rep
                
        elif type_ == "number":
            replacement = generate_random_number(original)
            
        if replacement and replacement != original:
            replacements[token.i] = replacement
            
            # Add definition
            # Prevent duplicate definitions
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
