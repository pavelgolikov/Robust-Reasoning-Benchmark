import os
import json
import glob
import shutil

def process_baseline_results(base_path):
    # base_path should be .../experiments/baseline/results
    print(f"Scanning {base_path} for JSON files...")
    
    # We expect path structure: .../results/<model>/<dataset>/*.json
    # We want to create .../results/<model>/<dataset>/compound/ and .../results/<model>/<dataset>/perturb/
    
    search_pattern = os.path.join(base_path, "*", "*", "*.json")
    json_files = glob.glob(search_pattern)
    
    moved_count = 0
    for file_path in json_files:
        # Avoid processing files that are already inside compound or perturb folders
        # by checking if their parent directory is 'compound' or 'perturb'
        parent_dir = os.path.basename(os.path.dirname(file_path))
        if parent_dir in ["compound", "perturb"]:
            continue
            
        dataset_dir = os.path.dirname(file_path)
        compound_dir = os.path.join(dataset_dir, "compound")
        perturb_dir = os.path.join(dataset_dir, "perturb")
        
        # Read the file to look for summary and temperature 0.6
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        is_compound = False
        
        # Check if it's a list (as evaluate_api.py outputs)
        if isinstance(data, list) and len(data) > 0:
            last_item = data[-1]
            if "summary" in last_item:
                summary = last_item["summary"]
                temp = summary.get("temperature")
                if isinstance(temp, float) and abs(temp - 0.6) < 1e-5:
                    is_compound = True
                elif isinstance(temp, (int, float)) and temp == 0.6:
                    is_compound = True
                    
        target_dir = compound_dir if is_compound else perturb_dir
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Move the file
        filename = os.path.basename(file_path)
        new_path = os.path.join(target_dir, filename)
        
        shutil.move(file_path, new_path)
        print(f"Moved {filename} -> {os.path.basename(target_dir)}/")
        moved_count += 1
        
    print(f"Successfully processed and moved {moved_count} files.")

if __name__ == "__main__":
    base_dir = "/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/baseline/results"
    process_baseline_results(base_dir)
