import re
import random

def apply_sentence_reversal(text, seed=None):
    if seed:
        random.seed(seed)
        
    # User requested simple reversal on periods.
    # This might split decimals or abbreviations (e.g. 3.14 -> 14.3), but satisfies the user's
    # request for "easy transformation there and back" given messy LaTeX.
    parts = text.split('.')
    parts = parts[::-1]
    return ".".join(parts)

def reverse_sentence_reversal(text):
    """
    Reverses the sentence reversal transformation.
    Applying reversal twice restores the original order.
    """
    return apply_sentence_reversal(text)
