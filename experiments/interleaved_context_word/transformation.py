import itertools

def apply_interleaved_context_word(problem_a, problem_b, seed=None):
    """
    Interleaves Problem A and Problem B word by word.
    First word A, second word B, third word A, etc.
    If lengths differ, the shorter one is repeated (cycled) to match the longer one.
    """
    # Flatten structure to words
    words_a = problem_a.split()
    words_b = problem_b.split()
    
    if not words_a: words_a = [""]
    if not words_b: words_b = [""]
    
    max_len = max(len(words_a), len(words_b))
    
    # Cycle shorter to match max_len
    input_a = list(itertools.islice(itertools.cycle(words_a), max_len))
    input_b = list(itertools.islice(itertools.cycle(words_b), max_len))
    
    # Interleave
    interleaved = []
    for a, b in zip(input_a, input_b):
        interleaved.append(a)
        interleaved.append(b)
        
    return " ".join(interleaved)

def reverse_interleaved_context_word(text):
    """
    Reverses the word-interleaved transformation.
    Extracts words at even indices (0, 2, 4...) which correspond to Problem A.
    Note: If Problem A was cycled during transformation, this will return the cycled (repeated) text.
    """
    words = text.split()
    # A is at 0, 2, 4...
    words_a = words[0::2]
    return " ".join(words_a)
