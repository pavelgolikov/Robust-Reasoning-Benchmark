import json
import os
import random

def normalize(text):
    if not text:
        return ""
    return "".join(text.split())

files = [
    'experiments/word_split_swap/results/GAIR_LIMO-v2_word_split_swap_s42_20260117_172124_raw.json',
    'experiments/wrappers/results/GAIR_LIMO-v2_wrappers_s42_20260117_172124_raw.json',
    'experiments/context_saturation/results/GAIR_LIMO-v2_context_saturation_s42_20260117_172124_raw.json',
    'experiments/interleaved_context_word/results/GAIR_LIMO-v2_interleaved_context_word_s42_20260117_172124_raw.json'
]

failures = []
for fpath in files:
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f: 
        data = json.load(f)
        exp = fpath.split('/')[1]
        for entry in data:
            norm_out = normalize(entry['output'])
            norm_orig = normalize(entry['unmodified_original'])
            
            # Simple containment check
            if norm_orig not in norm_out:
                failures.append({
                    'experiment': exp, 
                    'system_prompt': entry['system_prompt'], 
                    'ground_truth_decoded': entry['unmodified_original'], 
                    'model_output': entry['output']
                })

# Sample 30
sampled = failures[:30]
print(json.dumps(sampled, indent=2))
