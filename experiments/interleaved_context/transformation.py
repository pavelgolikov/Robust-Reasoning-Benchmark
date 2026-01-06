
import itertools

def chunk_string(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def apply_interleaved_context(problem_a, problem_b, chunk_size=60):
    """
    Interleaves chunks of problem_a and problem_b.
    Each chunk is `chunk_size` characters long.
    If one problem is shorter in total chunks, it repeats from the beginning.
    
    Structure:
    Chunk A1 (60 chars)
    Chunk B1 (60 chars)
    Chunk A2 (60 chars)
    Chunk B2 (60 chars)
    ...
    """
    # Flatten newlines to spaces to ensure consistent filling of 60 chars
    flat_a = problem_a.replace('\n', ' ').strip()
    flat_b = problem_b.replace('\n', ' ').strip()
    
    chunks_a = chunk_string(flat_a, chunk_size)
    chunks_b = chunk_string(flat_b, chunk_size)
    
    # Ensure at least one chunk exists
    if not chunks_a: chunks_a = [""]
    if not chunks_b: chunks_b = [""]
    
    max_len = max(len(chunks_a), len(chunks_b))
    
    # Extend shorter list by repeating
    extended_a = list(itertools.islice(itertools.cycle(chunks_a), max_len))
    extended_b = list(itertools.islice(itertools.cycle(chunks_b), max_len))
    
    interleaved = []
    for a, b in zip(extended_a, extended_b):
        interleaved.append(a)
        interleaved.append(b)
        
    return "\n".join(interleaved)
