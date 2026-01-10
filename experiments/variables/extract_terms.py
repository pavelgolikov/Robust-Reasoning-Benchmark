
import json
import os
import glob
import re

def extract_json_block(text):
    # Try different patterns for JSON code blocks
    patterns = [
        r"```json\s*(\{.*?\})\s*```",  # Standard markdown
        r"```\s*(\{.*?\})\s*```",      # Generic code block
        r"(\{.*?\})"                   # Raw JSON object (greedy)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Try to parse the last match (often the most refined one)
            for match in reversed(matches):
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
    return None

def main():
    result_dir = "experiments/variables/results"
    
    # Find latest file
    files = glob.glob(os.path.join(result_dir, "GAIR_LIMO-v2_variables_*.json"))
    if not files:
        print("No result files found in", result_dir)
        return
        
    latest_file = max(files, key=os.path.getmtime)
    print(f"Processing {latest_file}...")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
        
    extracted_by_problem = {}
    
    for item in data:
        prob_id = item.get('id')
        output_text = item.get('output', '')
        if not output_text:
            continue
            
        json_obj = extract_json_block(output_text)
        if json_obj:
            # Extract variables and entities
            variables = json_obj.get('variables', [])
            entities = json_obj.get('entities', [])
            
            terms = []
            for term in variables + entities:
                if isinstance(term, str):
                    clean_term = term.strip().replace('$', '').strip()
                    if clean_term:
                        terms.append(clean_term)
            
            if terms:
                # Dedup and sort
                extracted_by_problem[prob_id] = sorted(list(set(terms)))
    
    output_path = "experiments/variables/extracted_terms_by_problem.json"
    with open(output_path, 'w') as f:
        json.dump(extracted_by_problem, f, indent=2)
            
    print(f"Extracted terms for {len(extracted_by_problem)} problems to {output_path}")

if __name__ == "__main__":
    main()
