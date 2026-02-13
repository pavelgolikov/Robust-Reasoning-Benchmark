import spacy
import re

nlp = spacy.load("en_core_web_sm")

# Words that often start sentences in math problems but aren't names
MATH_STOPWORDS = {
    "Suppose", "Let", "Assume", "Given", "If", "When", "Then", 
    "Find", "Calculate", "Consider", "Where", "Since"
}

def extract_names_robust(problem_text):
    # 1. CLEAN: Remove LaTeX ($...$) and replace with a neutral placeholder
    # We replace with " " to prevent words merging (e.g. "end$start" -> "end start")
    clean_text = re.sub(r'\$[^$]+\$', ' ', problem_text)
    print(f"\nCleaned text: {clean_text}\n")
    
    # 2. PROCESS: Run spaCy
    doc = nlp(clean_text)
    
    names = set()
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            text = ent.text.strip()
            
            # 3. FILTER: Fix "Suppose Aya" -> "Aya"
            parts = text.split()
            
            # If the "Name" starts with a math stopword (e.g., "Suppose"), remove it
            if parts[0] in MATH_STOPWORDS:
                if len(parts) > 1:
                    # Keep the rest (e.g., "Suppose Aya" -> "Aya")
                    cleaned_name = " ".join(parts[1:])
                    names.add(cleaned_name)
                # If the entity is JUST "Suppose", ignore it entirely
                continue
            else:
                # 4. SANITY CHECK: Ensure no numbers/symbols remain
                if text.replace(" ", "").isalpha():
                    names.add(text)
                
    return list(names)

# --- TEST ---
problem = "Alice and Bob play the following game. A stack of $n$ tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either $1$ token or $4$ tokens from the stack. Whoever removes the last token wins. Find the number of positive integers $n$ less than or equal to $2024$ for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play."
print(problem)
print(f"Extracted: {extract_names_robust(problem)}")
