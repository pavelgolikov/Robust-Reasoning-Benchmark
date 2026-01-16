import json

file_path = 'experiments/context_saturation/results/GAIR_LIMO-v2_context_saturation_s42_20260115_045322.json'
output_file = 'experiments/context_saturation/failure_outputs.txt'

with open(file_path, 'r') as f:
    data = json.load(f)

failures = [entry for entry in data if not entry['correct']]

with open(output_file, 'w') as f:
    for i, fail in enumerate(failures[:3]):
        f.write(f"--- Failure {i+1} (ID: {fail['id']}) ---\n")
        f.write(f"Ground Truth: {fail['ground_truth']}\n")
        f.write("Output:\n")
        f.write(fail['output'])
        f.write("\n\n" + "="*80 + "\n\n")

print(f"Dumped {min(3, len(failures))} failures to {output_file}")
