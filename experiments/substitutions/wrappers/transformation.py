import spacy
import random

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

DISTRACTIVE_WRAPPERS = [
    "delete", "remove", "ignore", "skip", "omit", 
    "cancel", "void", "negate", "destroy", "erase",
    "discard", "drop", "exclude", "ban", "forbid",
    "stop", "halt", "abort", "reject", "deny",
    "refuse", "block", "prevent", "avoid", "evade",
    "miss", "fail", "break", "corrupt", "glitch",
    "bug", "error", "wrong", "false", "fake",
    "lie", "bad", "incorrect", "undefined", "null",
    "zero", "invalid", "unknown", "nill", "nothing",
    "empty", "blank", "vacant", "hollow", "noise"
]

def apply_wrapper(text, k=2, seed=None):
    """
    Identifies Nouns and Numbers.
    Randomly selects 1/k of them (default k=2 means 50%).
    Wraps them with a random distractive wrapper function.
    Adds definitions to defyn block.
    """
    if seed is not None:
        random.seed(seed)
        
    doc = nlp(text)
    
    candidates = []
    for token in doc:
        # NOUNs and NUMs
        if token.pos_ in ["NOUN", "NUM"] and token.text.isalnum():
            candidates.append(token)
            
    if not candidates:
        return text
        
    num_to_wrap = max(1, len(candidates) // k)
    to_wrap = random.sample(candidates, num_to_wrap)
    
    definitions = []
    replacements = {} 
    
    # We want to reuse wrappers for the same word if possible? 
    # Or just random wrapper every time? The prompt said "give me 50 potential names... List of wrapper names".
    # It implied using them. Let's pick a wrapper for each occurrence or each unique word?
    # "All the wrapper function calls will be identity"
    # "Let 'delete(x)' mean 'x'"
    # It's better to verify if we need to define the function generally `let "delete(x)" mean "x"` 
    # or per instance `let "delete(Apple)" mean "Apple"`.
    # Based on "opposites" implementation: `let "antonym" mean "original"`.
    # So here it should probably be: `let "wrapper(original)" mean "original"`.
    # Wait, the request said: "All the wrapper function calls will be identity."
    # If I say `let "delete(x)" mean "x"`, that's a general function definition.
    # But current infrastructure (as seen in README and opposites) uses `defyn { Let "A" mean "B" }`.
    # It seems to do simple string replacement definitions or specific instance definitions.
    # "opposites" does: `let "Replacement" mean "Original"`.
    # So I should do: `let "delete(5)" mean "5"`.
    
    used_wrappers = set()

    for token in to_wrap:
        word = token.text
        
        # Pick a random wrapper
        wrapper = random.choice(DISTRACTIVE_WRAPPERS)
        used_wrappers.add(wrapper)
        
        replacement = f"{wrapper}({word})"
        replacements[token.i] = replacement
        
        # Add definition
        # We need to escape definitions properly if needed, but for simple words it's fine.
        def_str = f'let "{replacement}" mean "{word}"'
        if def_str not in definitions:
            definitions.append(def_str)
            
    # Reconstruct
    output_tokens = []
    for i, token in enumerate(doc):
        if i in replacements:
            output_tokens.append(replacements[i] + token.whitespace_)
        else:
            output_tokens.append(token.text_with_ws)
            
    transformed_text = "".join(output_tokens)
    
    if definitions:
        def_block = "defyn{" + ", ".join(definitions) + "}.\n\n"
    else:
        def_block = ""
        
    return def_block + transformed_text

def get_distractive_wrappers_list():
    return DISTRACTIVE_WRAPPERS
