import re
import random

    
def apply_sentence_reversal(text, seed=None):
    if seed:
        random.seed(seed)
    parts = text.split('.')
    parts = parts[::-1]
    if parts[0] == '': # now the empty string might have been moved to the front - we simply move it back
        parts = parts[1:]
        parts.append('')
    ret = ".".join(parts)
    return ret


def reverse_sentence_reversal(text):
    """
    Reverses the sentence reversal transformation.
    Applying reversal twice restores the original order.
    """
    return apply_sentence_reversal(text)
