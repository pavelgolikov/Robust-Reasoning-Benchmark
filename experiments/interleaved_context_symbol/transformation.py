import itertools

def apply_interleaved_context_symbol(problem_a, problem_b, seed=None):
    """
    Interleaves Problem A and Problem B symbol by symbol (character by character).
    First symbol A, second symbol B, third symbol A, etc.
    If lengths differ, the shorter one is repeated (cycled) to match the longer one.
    """
    # Characters
    chars_a = list(problem_a)
    chars_b = list(problem_b)
    
    if not chars_a: chars_a = [""]
    if not chars_b: chars_b = [""]
    
    max_len = max(len(chars_a), len(chars_b))
    
    # Cycle shorter to match max_len
    input_a = list(itertools.islice(itertools.cycle(chars_a), max_len))
    input_b = list(itertools.islice(itertools.cycle(chars_b), max_len))
    
    # Interleave
    interleaved = []
    for a, b in zip(input_a, input_b):
        interleaved.append(a)
        interleaved.append(b)
        
    return "".join(interleaved)

def reverse_interleaved_context_symbol(text):
    """
    Reverses the symbol-interleaved transformation.
    Extracts symbols at even indices (0, 2, 4...) which correspond to Problem A.
    Note: If Problem A was cycled during transformation, this will return the cycled (repeated) text.
    """
    # A is at 0, 2, 4...
    # Direct slicing on string works for characters
    return text[0::2]
