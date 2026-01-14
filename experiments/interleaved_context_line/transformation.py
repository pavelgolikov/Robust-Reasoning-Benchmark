import itertools

def chunk_string(text, chunk_size):
    # Flatten newlines to spaces
    words = text.replace('\n', ' ').strip().split()
    lines = []
    current_line = []
    current_len = 0
    
    for word in words:
        # If we have content and adding another word (or just being over limit already)
        # The rule is: "if word goes over, just keep the word as is and break after it"
        # This implies we keep adding until we exceed 60, then the NEXT word starts a new line.
        
        if current_line and current_len >= chunk_size:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
        else:
            if current_line:
                current_len += 1 # space
            current_line.append(word)
            current_len += len(word)
            
    if current_line:
        lines.append(" ".join(current_line))
        
    return lines

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
    
    # 2. Tag chunks
    tagged_a = [c + " <Problem A>" for c in chunks_a]
    tagged_b = [c + " <Problem B>" for c in chunks_b]
    
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
    """
    lines = text.split('\n')
    reconstructed_parts = []
    
    for line in lines:
        if line.endswith(" <Problem A>"):
            # Remove tag (12 chars: " <Problem A>")
            content = line[:-12]
            reconstructed_parts.append(content)
            
    # Join with space as the original chunking flattened newlines to spaces
    return " ".join(reconstructed_parts)
