import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract unique variables from a JSON list of lists.")
    parser.add_argument("--input", "-i", type=str, default="AIME_2024_vars.json", help="Input JSON file path")
    parser.add_argument("--output", "-o", type=str, default="unique_vars.json", help="Output JSON file path")
    args = parser.parse_args()

    # Handle relative paths assumes relative to current working directory or absolute
    input_path = args.input
    if not os.path.exists(input_path):
        # Try looking one level up if in experiments folder
        parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.input)
        if os.path.exists(parent_path):
             input_path = parent_path
        else:
             print(f"Error: Input file '{args.input}' not found.")
             return

    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    if not isinstance(data, list):
         print("Error: JSON content is not a list.")
         return

    unique_vars = set()
    for item in data:
        if isinstance(item, list):
            for var in item:
                if isinstance(var, str):
                    unique_vars.add(var)
        elif isinstance(item, str):
             unique_vars.add(item)
    
    sorted_vars = sorted(list(unique_vars))
    
    print(f"Found {len(sorted_vars)} unique variables.")
    
    with open(args.output, 'w') as f:
        json.dump(sorted_vars, f, indent=2)
    
    print(f"Saved unique variables to {args.output}")
    print(f"Variables: {sorted_vars}")

if __name__ == "__main__":
    main()
