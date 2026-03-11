import spacy
import random
import re

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def _apply_not_not(text, k=2):
    """
    Inserts 'not not ' before first ADJ and every k-th eligible word (ADJ/NUM).
    Protects LaTeX content ($...$) from being transformed or mis-tagged.
    """
    # 1. Mask LaTeX blocks to protect them
    latex_blocks = []
    # Match $...$, $$...$$, \[...\], \(...\)
    pattern = r'(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\))'
    
    def mask_func(match):
        placeholder = f"MATHBLOCK{len(latex_blocks)}"
        latex_blocks.append(match.group(0))
        return placeholder

    masked_text = re.sub(pattern, mask_func, text, flags=re.DOTALL)
    
    doc = nlp(masked_text)
    
    eligible_indices = []
    first_adj_index = -1
    
    for i, token in enumerate(doc):
        t_text_lower = token.text.lower().strip()
        
        # RULE: Never insert 'not not' before 'such'
        if t_text_lower == "such":
            continue
            
        # RULE: Never insert 'not not' before a Protected Math Block
        if t_text_lower.startswith("mathblock"):
            continue

        if token.pos_ in ["ADJ", "NUM"]:
            eligible_indices.append(i)
            if first_adj_index == -1 and token.pos_ == "ADJ":
                first_adj_index = i
                
    insertion_indices = set()
    if first_adj_index != -1:
        insertion_indices.add(first_adj_index)
        
    for count, idx in enumerate(eligible_indices, 1):
        if count % k == 0:
            insertion_indices.add(idx)
            
    output_tokens = []
    for i, token in enumerate(doc):
        # 2. Check for insertion
        if i in insertion_indices:
            output_tokens.append("not not ")
            
        t_text = token.text
        # 3. Unmask LaTeX if this token is a placeholder
        if t_text.startswith("MATHBLOCK"):
            try:
                # Extract index from MATHBLOCKN
                idx = int(t_text[9:])
                original_math = latex_blocks[idx]
                # Preserve following whitespace from the original tokenization
                ws = token.text_with_ws[len(t_text):]
                output_tokens.append(original_math + ws)
            except (ValueError, IndexError):
                # Fallback to token text if something went wrong
                output_tokens.append(token.text_with_ws)
        else:
            output_tokens.append(token.text_with_ws)
        
    return "".join(output_tokens)

def apply_not_not(text, k=2, seed=None):
    """
    Applies 'not not' transformation.
    Inserts 'not not ' before first ADJ and every k-th eligible word.
    """
    if seed:
        random.seed(seed)
        
    return _apply_not_not(text, k=k)

def reverse_not_not(text):
    """
    Reverses the not-not transformation by removing 'not not '.
    """
    import re
    
    # Remove 'not not '
    # Note: Matches 'not not' followed by whitespace.
    text = re.sub(r'not\s+not\s+', '', text)
    
    return text


