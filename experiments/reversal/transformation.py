
import random

def reverse_string(text):
    return text[::-1]

TRANSFORMATIONS = {
    "reverse_string": reverse_string
}

def apply_reversal(text, separator=" ", func_name="reverse_string", seed=None):
    """
    Splits the text by 'separator', applies 'func_name' to each part,
    and joins them back with 'separator'.
    """
    if seed:
        random.seed(seed)
    
    if func_name not in TRANSFORMATIONS:
        raise ValueError(f"Unknown function: {func_name}. Available: {list(TRANSFORMATIONS.keys())}")
        
    func = TRANSFORMATIONS[func_name]
    
    if not separator:
        raise ValueError("Separator cannot be empty.")
        
    parts = text.split(separator)
    
    # Apply function to each part
    transformed_parts = [func(part) for part in parts]
    
    # Join back
    return separator.join(transformed_parts)

def reverse_reversal(text, separator=" ", func_name="reverse_string"):
    """
    Reverses the generalized reversal. 
    Assumes the transformation function is its own inverse (like reverse_string).
    """
    return apply_reversal(text, separator, func_name)
