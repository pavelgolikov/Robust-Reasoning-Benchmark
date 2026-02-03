import itertools

def chunk_string(text, chunk_size):
    # Flatten newlines to '; ' (semicolon + space)
    # This preserves line boundaries as clause separators
    flat = text.replace('\n', '; ').strip()
    return [flat[i:i+chunk_size] for i in range(0, len(flat), chunk_size)]

def apply_interleaved_context_line(problem_a, problem_b, seed=None):
    """
    Interleaves Problem A and Problem B (chunks of 60 chars).
    Adds tags <Problem A> / <Problem B>.
    Repeats shorter problem to match length of longer one.
    """
    # 1. Chunking
    chunk_size = 60
    chunks_a = chunk_string(problem_a, chunk_size)
    chunks_b = chunk_string(problem_b, chunk_size)
    
    # Ensure not empty
    if not chunks_a: chunks_a = [""]
    if not chunks_b: chunks_b = [""]
    
    # 2. Tag chunks (Prefix)
    tagged_a = ["<Problem A> " + c for c in chunks_a]
    tagged_b = ["<Problem B> " + c for c in chunks_b]
    
    max_len = max(len(tagged_a), len(tagged_b))
    
    # 3. Cycle shorter to match max_len
    input_a = list(itertools.islice(itertools.cycle(tagged_a), max_len))
    input_b = list(itertools.islice(itertools.cycle(tagged_b), max_len))
    
    # 4. Interleave
    interleaved_lines = []
    for a, b in zip(input_a, input_b):
        interleaved_lines.append(a)
        interleaved_lines.append(b)
        
    final_text = "\n".join(interleaved_lines)
    
    return final_text

def reverse_interleaved_context_line(text):
    """
    Reverses the interleaved_context transformation.
    Extracts lines tagged with <Problem A> and rejoins them.
    Assumes tags are prefixes: <Problem A> ...
    """
    lines = text.split('\n')
    reconstructed_parts = []
    
    for line in lines:
        if line.startswith("<Problem A> "):
            # Remove tag (12 chars: "<Problem A> ")
            content = line[12:]
            reconstructed_parts.append(content)
            
    # Join parts
    # Note: Newlines were replaced by '; ' in transformation.
    # We do NOT reverse this replacement here.
    full_text = "".join(reconstructed_parts)
    
    return full_text

