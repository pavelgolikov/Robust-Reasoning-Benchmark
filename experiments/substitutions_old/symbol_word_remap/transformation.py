import spacy
import random
import re

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

# Valid symbols and strings for remapping
FREQUENT_SYMBOLS_POOL = ['+', '-', '*', '/', '=', '<', '>', '^', '%', '(', ')', '[', ']', '{', '}', '&', '|', '!', '~', '→', '∀', '∃']

FREQUENT_STRINGS_POOL = [
    'variable', 'constant', 'sum', 'total', 'plus', 'add', 'combined', 'increase', 
    'minus', 'difference', 'subtract', 'less', 'reduce', 'decrease', 
    'times', 'product', 'multiply', 'double', 'triple', 'quadruple', 'twice', 
    'divide', 'quotient', 'split', 'per', 'ratio', 'fraction', 
    'square', 'cube', 'root', 'power', 'mean', 'remainder', 
    'if', 'then', 'else', 'otherwise', 'assume', 'suppose', 
    'and', 'or', 'not', 'nor', 'xor', 
    'therefore', 'hence', 'thus', 'implies', 'consequently', 'given', 
    'equals', 'is', 'equivalent', 'same', 
    'greater', 'larger', 'smaller', 'fewer', 'exceeds',
    'return', 'result', 'output', 'input', 'value', 'compute', 'calculate', 'solve', 'find', 'evaluate'
]

FORBIDDEN_MAPPINGS = {
    '+': ['increase', 'add', 'plus'],
    '-': ['decrease', 'subtract', 'less', 'minus'],
    '*': ['multiply', 'product'],
    '/': ['divide', 'quotient', 'per', 'ratio'],
    '%': ['remainder'],
    '=': ['equals', 'equivalent', 'same', 'is'],
    '<': ['less', 'fewer', 'smaller'],
    '>': ['greater', 'exceeds', 'more', 'larger'],
    '&': ['and'],
    '|': ['or'],
    '!': ['not']
}

def apply_symbol_remapping_transformation(text, k=3):
    """
    Remaps top k frequent symbols/strings in the text.
    Enforces Cross-Type Mapping:
    - Symbols -> Words
    - Words -> Symbols
    """
    doc = nlp(text)
    
    # Count Symbols
    symbol_counts = {}
    for sym in FREQUENT_SYMBOLS_POOL:
        count = text.count(sym)
        if count > 0:
            symbol_counts[sym] = count
            
    # Count Words (Dynamic)
    word_counts = {}
    for token in doc:
        if token.is_alpha and not token.is_stop and len(token.text) > 1:
            s_lower = token.text.lower()
            word_counts[s_lower] = word_counts.get(s_lower, 0) + 1

    # Select Targets
    # Top k symbols
    top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:k]
    # Top k words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # Prepare Replacements
    # We need k words for the symbols, and k symbols for the words.
    
    # Replacements for Symbols must be Words
    available_replacement_words = list(FREQUENT_STRINGS_POOL)
    
    # Replacements for Words must be Symbols
    available_replacement_symbols = [s for s in FREQUENT_SYMBOLS_POOL if s not in symbol_counts] 
    
    mappings = {}
    definitions = []
    
    # 1. Map Symbols -> Words
    for sym, count in top_symbols:
        forbidden = FORBIDDEN_MAPPINGS.get(sym, [])
        candidates = [s for s in available_replacement_words if s not in forbidden]
        
        if candidates:
            replacement_word = random.choice(candidates)
            mappings[sym] = replacement_word
            available_replacement_words.remove(replacement_word)
            definitions.append(f'let "{replacement_word}" mean "{sym}"')
            
    # 2. Map Words -> Symbols
    for word_lower, count in top_words:
        # Find original casing for replacement logic
        original_word = word_lower
        for token in doc:
            if token.text.lower() == word_lower:
                original_word = token.text
                break
                
        if available_replacement_symbols:
            replacement_sym = random.choice(available_replacement_symbols)
            mappings[word_lower] = replacement_sym
            available_replacement_symbols.remove(replacement_sym)
            definitions.append(f'let "{replacement_sym}" mean "{original_word}"')

    # Apply Swaps
    transformed_text = text
    # Sort by length to avoid partial replacements (though words vs symbols usually distinct)
    sorted_map = sorted(mappings.items(), key=lambda x: len(x[0]), reverse=True)
    
    for target, replacement in sorted_map:
        # Check if target is a word (length > 1 and alpha-ish) or symbol
        if len(target) > 1 and target[0].isalpha():
            # It's a word target -> Regex replacement
            pattern = re.compile(r'\b' + re.escape(target) + r'\b', re.IGNORECASE)
            transformed_text = pattern.sub(replacement, transformed_text)
        else:
            # It's a symbol target -> String replace
            transformed_text = transformed_text.replace(target, replacement)
            
    if definitions:
        def_block = "defyn{" + ", ".join(definitions) + "}.\n\n"
    else:
        def_block = ""
        
    return def_block + transformed_text
