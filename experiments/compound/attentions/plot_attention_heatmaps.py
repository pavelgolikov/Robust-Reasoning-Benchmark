import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_file(filepath):
    print(f"Parsing {filepath}...")
    layer_data = defaultdict(lambda: {
        'System': [], 
        'Distractor': [], 
        'Target': []
    })
    
    layer_pattern = re.compile(
        r"Layer\s+(\d+)\s+Averages\s+-\s+System:\s+([0-9.]+)%,\s+DistractorPre:\s+([0-9.]+)%,\s+TargetStmt:\s+([0-9.]+)%,\s+DistractorPost:\s+([0-9.]+)%,\s+Target:\s+([0-9.]+)%"
    )
    
    in_sample = False
    
    with open(filepath, 'r') as f:
        for line in f:
            if "=== SAMPLE" in line:
                in_sample = True
            elif "=== AGGREGATED" in line:
                in_sample = False
                
            if in_sample:
                match = layer_pattern.search(line)
                if match:
                    layer_idx = int(match.group(1))
                    sys_val = float(match.group(2))
                    dpre_val = float(match.group(3))
                    tstmt_val = float(match.group(4))
                    dpost_val = float(match.group(5))
                    tgt_val = float(match.group(6))
                    
                    layer_data[layer_idx]['System'].append(sys_val)
                    layer_data[layer_idx]['Distractor'].append(dpre_val + dpost_val)
                    layer_data[layer_idx]['Target'].append(tstmt_val + tgt_val)
                    
    return layer_data

def generate_heatmap(layer_data, model_name, date_time, out_path):
    if not layer_data:
        print(f"No data found for {model_name}. Skipping plot.")
        return
        
    layers = sorted(layer_data.keys())
    num_layers = len(layers)
    
    regions = ['System', 'Distractor', 'Target']
    
    data_matrix = np.zeros((len(regions), num_layers))
    
    for i, region in enumerate(regions):
        for j, layer in enumerate(layers):
            data_matrix[i, j] = np.mean(layer_data[layer][region])
            
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    
    # Use coolwarm colormap as requested with matplotlib
    cax = ax.imshow(data_matrix, cmap="viridis", vmin=0, vmax=100, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('Attention Mass (%)')
    
    # Set ticks
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(regions)))
    
    # Set tick labels
    ax.set_xticklabels(layers)
    ax.set_yticklabels(regions)

    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    
    plt.title(f"Attention Dilution: {model_name}\n(Run Date: {date_time})", fontsize=14, pad=15)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Prompt Region", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    target_dir = "/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/compound/attentions"
    
    # Target files for Qwen and Nemotron (7B and 32B)
    pattern = os.path.join(target_dir, "dilution_*_3distractors_*.txt")
    files = glob.glob(pattern)
    
    target_files = [f for f in files if "Qwen" in f or "Nemotron" in f]
    
    for filepath in target_files:
        basename = os.path.basename(filepath)
        # Parse model name and date from filename
        # Expected format: dilution_{model_name}_{distractors}distractors_{date}_{time}.txt
        match = re.search(r"dilution_(.+?)_\d+distractors_(\d{8}_\d{6})\.txt", basename)
        if match:
            model_name = match.group(1)
            date_time = match.group(2)
        else:
            model_name = basename.replace("dilution_", "").split("_3distractors")[0]
            date_time = "Unknown"
            
        layer_data = parse_file(filepath)
        
        # Output PDF path
        out_filename = f"heatmap_{model_name}_{date_time}.pdf"
        out_path = os.path.join(target_dir, out_filename)
        
        generate_heatmap(layer_data, model_name, date_time, out_path)

if __name__ == "__main__":
    main()
