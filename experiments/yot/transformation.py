import spacy
import random

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def apply_yot_transformation(text, k=3):
    """
    Inserts 'yot', 'yot yot', or 'not not' before every adjective and number.
    'yot' is defined as the opposite of 'not' (identity).
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
            # Randomly select a phrase
            phrase = random.choice(options)
            output_tokens.append(phrase + " ")
        output_tokens.append(token.text_with_ws)
        
    return "".join(output_tokens)
