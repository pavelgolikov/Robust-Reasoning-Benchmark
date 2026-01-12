import spacy
import random

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def apply_sentence_reversal(text, seed=None):
    if seed:
        random.seed(seed)
        
    doc = nlp(text)
    
    # Get sentences with their trailing whitespace
    sentences = [sent.text_with_ws for sent in doc.sents]
    
    # Reverse
    reversed_sentences = sentences[::-1]
    
    return "".join(reversed_sentences)
