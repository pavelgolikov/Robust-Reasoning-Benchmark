
import json
import argparse
import sys
import os

# Add local directory to path to import util
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util import verify_answer, extract_answer

def reverify_file(filepath):
    print(f"Processing {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

    # Handle both list and dict formats
    if isinstance(data, list):
        results = data
        is_list_format = True
    else:
        results = data.get('results', [])
        is_list_format = False

    stats = {"correct": 0, "total": 0, "failures": 0}
    
    changes = 0
    
    for res in results:
        extracted = res.get('extracted')
        # If extracted is None, try to extract
        if extracted is None:
             extracted = extract_answer(res.get('output', ''))
             res['extracted'] = extracted

        ground_truth = res.get('ground_truth')
        
        # New verification
        is_correct = False
        if extracted:
            is_correct = verify_answer(extracted, ground_truth)
        else:
            # Try once more to extract if verify_answer relies on it
            # But we already tried above.
            pass
        
        # Update stats
        stats['total'] += 1
        if is_correct:
            stats['correct'] += 1
        else:
            stats['failures'] += 1
            
        # Check if status changed
        old_correct = res.get('correct', False)
        if old_correct != is_correct:
            changes += 1
        
        res['correct'] = is_correct

    # Update stats in data if it's a dict, otherwise print them
    if not is_list_format:
        data['statistics'] = stats
    
    # Save back
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Updated {filepath}")
    print(f"Total: {stats['total']}, Correct: {stats['correct']}, Accuracy: {stats['correct']/stats['total'] if stats['total'] > 0 else 0:.2%}")
    print(f"Status changed for {changes} examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="JSON result files to reverify")
    args = parser.parse_args()
    
    for f in args.files:
        reverify_file(f)
