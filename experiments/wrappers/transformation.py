
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
            
        original_text = token.text
        
        # Regex for integers
        text_1 = re.sub(r'\d+', wrap_word, original_text)
        
        def wrap_var(match):
            w = match.group()
            if w.lower() == 'a': return w # Skip 'a'
            return wrap_word(match)
            
        text_2 = re.sub(r'(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])', wrap_var, text_1)
        
        output_tokens.append(text_2 + token.whitespace_)

    transformed_text = "".join(output_tokens)
    
    if definitions:
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
    if not mappings:
        return text_clean

    # Sort keys by length descending to prioritize longer matches
    sorted_keys = sorted(mappings.keys(), key=len, reverse=True)
    
    # Escape keys for regex
    escaped_keys = [re.escape(k) for k in sorted_keys]
    
    # Compile master regex
    master_pattern = re.compile('|'.join(escaped_keys))
    
    def replace_callback(m):
        return mappings[m.group(0)]
        
    text_clean = master_pattern.sub(replace_callback, text_clean)
        
    # 5. Clean up excessive whitespace
    text_clean = re.sub(r'\n{3,}', '\n\n', text_clean)
    
    return text_clean.strip()

