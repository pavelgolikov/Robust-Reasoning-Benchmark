
import spacy
import random
import re

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def apply_wrappers(text, k=1, seed=None):
    if seed is not None:
        random.seed(seed)
        
    doc = nlp(text)
    
    # Store replacements: index -> formatting logic or replacement text
    # But since we might make multiple decisions per token, we can't pre-calculate easily without doing it live?
    # We can pre-calculate candidates, but for complex tokens it's easier to just build the replacement string.
    
    # We will build "final tokens" list directly.
    
    # Global definitions list
    definitions = []
    used_numbers = set()
    
    def get_wrapper(word):
        # Range is small (2-10), so we might exhaust unique numbers.
        # Fallback to reuse if needed.
        start, end = 2, 10
        total_possible = end - start + 1
        
        if len(used_numbers) >= total_possible:
             return str(random.randint(start, end))
             
        while True:
            wn = str(random.randint(start, end))
            if wn not in used_numbers:
                used_numbers.add(wn)
                return wn
                
    def wrap_word(match):
        word = match.group()
        # Decision: do we wrap this specific word?
        # To respect k parameter strictly is hard if we do it on the fly.
        # But if k=1 (default for this check), we wrap everything.
        # For k>1, we might skip.
        
        # Simple sampling: chance = 1/k
        if k > 1 and random.random() > (1.0/k):
            return word
            
        wn = get_wrapper(word)
        def_str = f'let "{wn}({word})" mean "{word}"'
        definitions.append(def_str)
        return f"{wn}({word})"

    output_tokens = []
    
    for token in doc:
        # 1. Clean Nouns/Numbers/Propn
        if token.pos_ in ["NOUN", "NUM", "PROPN"] and token.text.isalnum():
            # Apply sampling
            if k == 1 or random.random() <= (1.0/k):
                wn = get_wrapper(token.text)
                def_str = f'let "{wn}({token.text})" mean "{token.text}"'
                definitions.append(def_str)
                output_tokens.append(f"{wn}({token.text})" + token.whitespace_)
            else:
                output_tokens.append(token.text_with_ws)
            continue
            
        # 2. Complex tokens
        # We want to wrap numbers (\d+) and single letter variables ([a-zA-Z]) inside them.
        # Spacy definition of token generally separates words, so "kilometer" inside "9$-kilometer" 
        # is actually unlikely to be a "word" in non-symbol sense if stuck to $.
        # Regex to find wrap-able chunks:
        # Numbers: \d+
        # Single Variables: \b[a-zA-Z]\b ? No, "s" in "s+2" is \b. 
        # But "s" in "steps" is not variable.
        # We should capture single letters that are surrounded by non-letters?
        # Or just numbers for now to satisfy "9".
        # User said: "wrap variables... e.g. B and C... and 9".
        
        # Pattern: Numbers OR Single Letters (Variables)
        # Note: Be careful not to wrap "a" in "parameter". 
        # But inside a single token like "s+2", "parameter" isn't there.
        # If token is "steps", single letter 's' is at start/end.
        # Complex token implies it contains symbols.
        # If token text is just "steps", it fell through (unless pos is NOUN?). "steps" is NOUN.
        # So "steps" was handled in block 1 (if isalnum).
        
        # If we are here, token is NOT isalnum or NOT target POS.
        # e.g. "s+2" (NOUN?), "$9$-kilometer" (NUM).
        
        # We apply regex replacement for digits and single-letter-vars.
        original_text = token.text
        
        # Regex for integers
        # We use a lambda to wrap
        text_1 = re.sub(r'\d+', wrap_word, original_text)
        
        # Regex for single letters that look like variables?
        # e.g. 's' in 's+2'. 't' in '$t$'.
        # We generally want to avoid 'a' (article).
        # And avoid parts of words like 'k' in 'kilometer' if 'kilometer' is part of the token?
        # In '9$-kilometer', 'k', 'i', 'l'...
        # 'kilometer' is a word.
        # If we just target \d+, we solve the '9' problem.
        # If we target single letters surrounded by non-word chars?
        # (?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])
        # This handles 's' in 's+2'.
        # This handles 't' in '$t$'.
        # This ignores 'k' in 'kilometer'.
        
        def wrap_var(match):
            w = match.group()
            if w.lower() == 'a': return w # Skip 'a'
            return wrap_word(match)
            
        text_2 = re.sub(r'(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])', wrap_var, text_1)
        
        output_tokens.append(text_2 + token.whitespace_)

    transformed_text = "".join(output_tokens)
    
    if definitions:
        # Deduplicate definitions?
        # My logic appends for every instance.
        # Unique definitions only to keep block clean?
        # "let 123(x) mean x".
        # If I wrap x twice with 123, duplicate def is fine but redundant.
        # If I wrap x twice with different numbers (random behavior), I need both defs.
        # Since I generate random num every time, keep all defs.
        
        # Format block
        def_block = "defyn{" + ", ".join(definitions) + "}."
        
        mid = len(transformed_text) // 2
        left_mid = transformed_text.rfind(' ', 0, mid)
        right_mid = transformed_text.find(' ', mid)
        
        if left_mid == -1: split_idx = right_mid
        elif right_mid == -1: split_idx = left_mid
        else:
            if (mid - left_mid) < (right_mid - mid): split_idx = left_mid
            else: split_idx = right_mid
        if split_idx == -1: split_idx = mid
             
        top_half = transformed_text[:split_idx]
        bottom_half = transformed_text[split_idx:]
        
        final_text = top_half + "\n\n" + def_block + "\n\n" + bottom_half
    else:
        final_text = transformed_text
        
    return final_text

def reverse_wrappers(text):
    """
    Reverses the wrapper transformation by parsing the defyn block 
    and unwrapping the defined terms.
    """
    import re
    
    # 1. Find and Extract defyn block
    # Pattern: defyn{ ... }.
    # Using DOTALL to match across newlines (though usually it's one line)
    
    pattern = re.compile(r'defyn\{(.*?)\}\.', re.DOTALL)
    match = pattern.search(text)
    
    if not match:
        return text
    
    def_content = match.group(1)
    
    # 2. Parse definitions
    def_pattern = re.compile(r'let "(.*?)" mean "(.*?)"')
    mappings = {}
    
    for m in def_pattern.finditer(def_content):
        wrapped_form = m.group(1) # e.g. "8(apple)"
        original_form = m.group(2) # e.g. "apple"
        mappings[wrapped_form] = original_form
        
    # 3. Remove the defyn block
    text_clean = text.replace(match.group(0), "")
    
    # 4. Apply replacements
    # Sort by length descending to handle potential overlaps
    sorted_keys = sorted(mappings.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        val = mappings[key]
        text_clean = text_clean.replace(key, val)
        
    # 5. Clean up excessive whitespace
    text_clean = re.sub(r'\n{3,}', '\n\n', text_clean)
    
    return text_clean.strip()

