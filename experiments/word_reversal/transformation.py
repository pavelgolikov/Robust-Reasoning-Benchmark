import spacy
import random

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit(1)

def apply_word_reversal(text, seed=None):
    if seed:
        random.seed(seed)
        
    import re
    doc = nlp(text)
    output_pieces = []
    
    # Process each sentence separately
    for sent in doc.sents:
        # Get raw text of sentence including trailing whitespace from last token?
        # sent.text usually includes it if tokens include it? No.
        # sent.text_with_ws includes trailing ws of sentence.
        raw_sent = sent.text_with_ws
        
        # Tokenizer Logic:
        # 1. Math Blocks ($...$) - greedy inner
        # 2. Whitespace
        # 3. Everything else (potential words)
        
        # We want to capture groups.
        # r'(\$[^\$]+\$)|(\s+)|([^\s\$]+)'
        
        tokens = []
        matches = re.finditer(r'(\$[^\$]+\$)|(\s+)|([^\s\$]+)', raw_sent)
        
        # Structure: list of token objects or just strings?
        # We need to identify which are "reversable words".
        # Reversable: Math blocks, regular words (stripping punct).
        # Irreversible: Whitespace, Punctuation.
        
        parsed_tokens = []
        words_to_reverse = [] # This stores the CONTENT of words to be swapped.
        word_indices = []     # Indices in parsed_tokens that are word-slots.
        
        for m in matches:
            math, space, other = m.groups()
            
            if math:
                # Math block is a "word" that moves.
                parsed_tokens.append({"type": "word_slot", "punct": ""})
                words_to_reverse.append(math)
                word_indices.append(len(parsed_tokens) - 1)
            elif space:
                parsed_tokens.append({"type": "space", "content": space})
            elif other:
                # "other" is a sequence of non-space chars. e.g. "Hello," or "world."
                # We separate trailing punctuation.
                # Regex: (WordContent)(Punctuation)$ 
                # Be careful of "Mr." -> "Mr" + "." ? User asked for "Punctuation marks remain".
                # "stops." -> "stops" + "."
                # "world," -> "world" + ","
                # Also include closing brackets/parens as punctuation to avoid moving them?
                pm = re.match(r'^(.*?)([,.;:?!\]\)}]*)$', other)
                if pm:
                    content, punct = pm.groups()
                else:
                    content, punct = other, ""
                    
                if not content:
                    # Just punctuation? e.g. "..." -> content="", punct="..."
                    # Treat as fixed punctuation.
                    parsed_tokens.append({"type": "fixed", "content": punct})
                else:
                    # It's a word.
                    parsed_tokens.append({"type": "word_slot", "punct": punct})
                    words_to_reverse.append(content)
                    word_indices.append(len(parsed_tokens) - 1)
        
        # Reverse the collected words
        reversed_words = words_to_reverse[::-1]
        
        # Reconstruct sentence
        sent_str = ""
        rw_idx = 0
        
        for pt in parsed_tokens:
            if pt["type"] == "space" or pt["type"] == "fixed":
                sent_str += pt["content"]
            elif pt["type"] == "word_slot":
                # Fill with next reversed word + original trailing punct
                sent_str += reversed_words[rw_idx] + pt["punct"]
                rw_idx += 1
                
        output_pieces.append(sent_str)
        
    return "".join(output_pieces)

def reverse_word_reversal(text):
    """
    Reverses the word reversal transformation.
    Since the transformation is symmetric (reversing a list twice yields the original),
    we can simply re-apply the transformation logic.
    """
    return apply_word_reversal(text)
