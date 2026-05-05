import os
import glob
import json
import multiprocessing
from functools import partial
from util import extract_and_grade
from tqdm import tqdm

def process_file(filepath, base_dir):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception as e:
        return {"error": f"Error loading {filepath}: {e}"}
        
    items = data.get('results', data) if isinstance(data, dict) else data
    
    if not isinstance(items, list):
        return {"error": f"Skipping {filepath} - unexpected format"}
        
    old_correct = 0
    new_correct = 0
    file_questions = len(items)
    
    for item in items:
        if 'output' not in item or 'ground_truth' not in item:
            continue
            
        orig_correct = item.get('correct', False)
        if orig_correct:
            old_correct += 1
            
        try:
            new_extracted, is_correct = extract_and_grade(item['output'], item['ground_truth'], exp_name='compound')
            if is_correct:
                new_correct += 1
        except Exception:
            # Fallback if parsing fails catastrophically
            pass
            
    return {
        "filepath": filepath,
        "rel_path": os.path.relpath(filepath, base_dir),
        "old_correct": old_correct,
        "new_correct": new_correct,
        "file_questions": file_questions,
        "changed": old_correct != new_correct
    }

def main():
    base_dir = "/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/compound/results"
    output_report = "/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/compound_regrade_report.txt"
    
    json_files = glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)
    
    print(f"Found {len(json_files)} json files. Starting regrading...")
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    func = partial(process_file, base_dir=base_dir)
    
    results = []
    for res in tqdm(pool.imap_unordered(func, json_files), total=len(json_files)):
        results.append(res)
        
    pool.close()
    pool.join()
    
    report_lines = []
    report_lines.append(f"Regrading Report for {len(json_files)} files\n")
    report_lines.append("="*80)
    
    total_old_correct = 0
    total_new_correct = 0
    total_questions = 0
    files_with_changes = 0
    
    for res in sorted(results, key=lambda x: x.get('rel_path', '')):
        if "error" in res:
            report_lines.append(res["error"])
            continue
            
        total_old_correct += res["old_correct"]
        total_new_correct += res["new_correct"]
        total_questions += res["file_questions"]
        
        if res["changed"]:
            files_with_changes += 1
            report_lines.append(f"File: {res['rel_path']}")
            report_lines.append(f"  Old Accuracy: {res['old_correct']}/{res['file_questions']} ({(res['old_correct']/max(1,res['file_questions']))*100:.2f}%)")
            report_lines.append(f"  New Accuracy: {res['new_correct']}/{res['file_questions']} ({(res['new_correct']/max(1,res['file_questions']))*100:.2f}%)\n")
            
    report_lines.append("="*80)
    report_lines.append(f"Total files with changes: {files_with_changes}")
    report_lines.append(f"Global Old Accuracy: {total_old_correct}/{total_questions} ({(total_old_correct/max(1,total_questions))*100:.2f}%)")
    report_lines.append(f"Global New Accuracy: {total_new_correct}/{total_questions} ({(total_new_correct/max(1,total_questions))*100:.2f}%)")
    
    with open(output_report, "w") as f:
        f.write("\n".join(report_lines) + "\n")
        
    print(f"Regrading complete. Report saved to {output_report}")

if __name__ == '__main__':
    main()
