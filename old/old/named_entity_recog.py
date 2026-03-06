import spacy
import re
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

# Perform load in global scope but handle potential errors gracefully if models missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please install it.")
    exit(1)

# Words that often start sentences in math problems but aren't names
MATH_STOPWORDS = {
    "Suppose", "Let", "Assume", "Given", "If", "When", "Then", 
    "Find", "Calculate", "Consider", "Where", "Since"
}

def extract_names_robust(problem_text):
    # 1. CLEAN: Remove LaTeX ($...$) and replace with a neutral placeholder
    # We replace with " " to prevent words merging (e.g. "end$start" -> "end start")
    clean_text = re.sub(r'\$[^$]+\$', ' ', problem_text)
    
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
                    # Double check if the cleaned name is just another stopword
                    if cleaned_name in MATH_STOPWORDS:
                        continue
                    names.add(cleaned_name)
                # If the entity is JUST "Suppose", ignore it entirely
                continue
            else:
                # 4. SANITY CHECK: Ensure no numbers/symbols remain
                # Allow simple spaces (e.g. "Mary Ann")
                if text.replace(" ", "").isalpha():
                    names.add(text)
                
    return list(names)

def main():
    parser = argparse.ArgumentParser(description="Extract named entities from a dataset (math problems).")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--output", type=str, default="extracted_names.json", help="Output JSON file for names")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset} (split={args.split})...")
    try:
        dataset = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} examples.")
        
    all_names = set()
    
    print("Extracting names...")
    # Using tqdm for progress bar
    for example in tqdm(dataset):
        # Support different column names - usually 'problem' or 'question'
        # Some datasets have 'question', others 'problem'
        text = example.get('problem', example.get('question', ''))
        if not text and 'question' in example:
             text = example['question']
             
        if not text:
            # Fallback for other structures?
            # Print available keys if first time failing
            continue
            
        names = extract_names_robust(text)
        all_names.update(names)
        
    print(f"\nTotal unique names found: {len(all_names)}")
    sorted_names = sorted(list(all_names))
    print(f"Names: {sorted_names}")
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(sorted_names, f, indent=2)
    print(f"Saved list of names to {args.output}")

if __name__ == "__main__":
    main()
