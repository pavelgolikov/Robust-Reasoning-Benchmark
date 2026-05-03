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

def get_matrix(layer_data, exclude_system=False):
    layers = sorted(layer_data.keys())
    num_layers = len(layers)
    
    if exclude_system:
        regions = ['Distractor', 'Target']
        data_matrix = np.zeros((len(regions), num_layers))
        for j, layer in enumerate(layers):
            d_val = np.mean(layer_data[layer]['Distractor'])
            t_val = np.mean(layer_data[layer]['Target'])
            total = d_val + t_val
            if total > 0:
                data_matrix[0, j] = (d_val / total) * 100.0
                data_matrix[1, j] = (t_val / total) * 100.0
    else:
        regions = ['System', 'Distractor', 'Target']
        data_matrix = np.zeros((len(regions), num_layers))
        for i, region in enumerate(regions):
            for j, layer in enumerate(layers):
                data_matrix[i, j] = np.mean(layer_data[layer][region])
    
    # Calculate average across all layers for each region and append as last column
    layer_averages = np.mean(data_matrix, axis=1, keepdims=True)
    data_matrix = np.hstack((data_matrix, layer_averages))
    x_labels = [str(l) for l in layers] + ['Avg']
    
    return data_matrix, x_labels, regions

def generate_combined_heatmap(all_results, out_path, exclude_system=False):
    num_models = len(all_results)
    if num_models == 0:
        return

    fig, axes = plt.subplots(num_models, 1, figsize=(16, 3.0 * num_models), squeeze=False)
    
    title = "Attention Dilution across Layers"
    if exclude_system:
        title += " (Excluding System Context)"
    fig.suptitle(title, fontsize=24, y=0.98)

    for idx, (layer_data, model_name) in enumerate(all_results):
        ax = axes[idx, 0]
        data_matrix, x_labels, regions = get_matrix(layer_data, exclude_system)
        
        cax = ax.imshow(data_matrix, cmap="viridis", vmin=0, vmax=100, aspect='auto')
        
        # Add text to the 'Avg' column
        avg_col_idx = data_matrix.shape[1] - 1
        for i in range(data_matrix.shape[0]):
            val = data_matrix[i, avg_col_idx]
            text_color = "white"
            ax.text(avg_col_idx, i, f"{val:.1f}     ", ha="center", va="center", color=text_color, fontweight="bold", fontsize=16)

        ax.set_title(model_name, fontsize=20, pad=10)
        # Show every 4th layer index + the 'Avg' column
        tick_indices = list(range(0, len(x_labels) - 1, 4))
        if (len(x_labels) - 2) not in tick_indices: # Ensure we show something near the end if needed, but 'Avg' is always last
             pass
        tick_indices.append(len(x_labels) - 1)
        tick_labels = [x_labels[i] for i in tick_indices]
        
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=18)
        ax.set_yticks(np.arange(len(regions)))
        ax.set_yticklabels(regions, fontsize=16)
        # ax.set_ylabel("Context Region", fontsize=18)
        
        if idx == num_models - 1:
            ax.set_xlabel("Layer index", fontsize=18)

    # Add a single colorbar for the whole figure
    fig.subplots_adjust(right=0.85, hspace=0.7)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axes[0, 0].images[0], cax=cbar_ax)
    cbar.set_label('Attention Mass (%)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {out_path}")

def main():
    target_dir = "/home/golikovp/Antigravity/Robust-Reasoning-Benchmark/experiments/compound/attentions"
    
    # Target patterns for Qwen and Nemotron
    patterns = [
        os.path.join(target_dir, "dilution_Qwen_*_3distractors_*.txt"),
        os.path.join(target_dir, "dilution_nvidia_OpenReasoning-Nemotron-7B_3distractors_*.txt"),
        os.path.join(target_dir, "dilution_nvidia_OpenReasoning-Nemotron-32B_3distractors_*.txt")
    ]
    
    all_results = []
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if not files:
            continue
        # Use the latest file if multiple exist
        latest_file = max(files, key=os.path.getmtime)
        
        basename = os.path.basename(latest_file)
        match = re.search(r"dilution_(.+?)_\d+distractors_", basename)
        model_name = match.group(1) if match else basename
        model_name = model_name.replace("nvidia_", "").replace("Qwen_", "")
        
        layer_data = parse_file(latest_file)
        all_results.append((layer_data, model_name))
    
    # Generate combined plots
    generate_combined_heatmap(all_results, os.path.join(target_dir, "combined_attention_dilution.pdf"), exclude_system=False)
    generate_combined_heatmap(all_results, os.path.join(target_dir, "combined_attention_dilution_no_system.pdf"), exclude_system=True)

if __name__ == "__main__":
    main()
