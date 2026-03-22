import os
import glob
import json
import argparse
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
import numpy as np

def get_exact_token_counter(model_name):
    lower_model = model_name.lower()
    if 'gpt' in lower_model or 'o1' in lower_model:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    elif 'claude' in lower_model:
        import anthropic
        client = anthropic.Anthropic()
        return lambda text: client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model=model_name,
            messages=[{"role": "user", "content": text}]
        ).input_tokens
    elif 'gemini' in lower_model:
        from google import genai
        client = genai.Client()
        return lambda text: client.models.count_tokens(model=model_name, contents=text).total_tokens
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return lambda text: len(tokenizer.encode(text))

def main():
    parser = argparse.ArgumentParser(description="Analyze baseline difficulty and output lengths.")
    parser.add_argument("--base_dir", type=str, default="experiments/baseline/results", help="Path to baseline results")
    parser.add_argument("--out_dir", type=str, default="experiments/analysis/plots", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Find all JSON files in the baseline results directory
    json_files = glob.glob(os.path.join(args.base_dir, "**", "*.json"), recursive=True)
    if not json_files:
        print(f"No JSON files found in {args.base_dir}")
        return

    # Dictionary to store stats per problem ID
    # problem_stats[id] = {'failures': int, 'total': int, 'lengths': list}
    problem_stats = defaultdict(lambda: {'failures': 0, 'total': 0, 'lengths': []})
    
    # Cache for tokenizers to avoid reloading the same one
    tokenizer_cache = {}

    for filepath in json_files:
        print(f"Processing {filepath}...")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Failed to read {filepath}: {e}")
            continue
            
        results = data if isinstance(data, list) else data.get("results", [])
        if not results:
            continue
            
        # Extract summary block if present at the end
        summary_block = None
        if results and "summary" in results[-1]:
            summary_block = results.pop(-1)
            
        # Determine model name
        model_name = None
        if summary_block:
            model_name = summary_block['summary'].get('model', None)
            
        if not model_name:
            # Infer from path: results/<model_name>/<dataset>/run_id.json
            parts = filepath.replace('\\', '/').split('/')
            try:
                results_idx = parts.index('results')
                model_name = parts[results_idx + 1]
                # Fix slashes for Hugging Face models if replaced with underscores
                if "_" in model_name and not any(model_name.startswith(p) for p in ["gpt", "claude", "gemini"]):
                    model_name = model_name.replace('_', '/', 1)
            except (ValueError, IndexError):
                print(f"  Could not infer model name from {filepath}")
                continue

        # Load tokenizer if not already loaded
        if model_name not in tokenizer_cache:
            print(f"  Loading exact tokenizer for model {model_name}...")
            try:
                tokenizer_cache[model_name] = get_exact_token_counter(model_name)
            except Exception as e:
                print(f"  Failed to load tokenizer for {model_name}: {e}. Skipping parsing lengths for this file.")
                tokenizer_cache[model_name] = None
        
        token_counter = tokenizer_cache[model_name]

        for entry in results:
            if "id" not in entry:
                continue
                
            problem_id = str(entry["id"])
            is_correct = entry.get("correct", False)
            output = entry.get("output", "")
            
            # Update correct/failure stats
            problem_stats[problem_id]['total'] += 1
            if not is_correct:
                problem_stats[problem_id]['failures'] += 1
                
            # Compute lengths
            if "output_tokens" in entry:
                problem_stats[problem_id]['lengths'].append(entry["output_tokens"])
            elif output and token_counter:
                try:
                    count = token_counter(output)
                    problem_stats[problem_id]['lengths'].append(count)
                except Exception as e:
                    print(f"  Failed to count tokens for problem {problem_id}: {e}")

    if not problem_stats:
        print("No valid problem statistics aggregated.")
        return

    # Prepare data for plotting
    problem_ids = list(problem_stats.keys())
    
    # 1. Sort by difficulty (failures)
    # We plot raw failures as requested: "most failures across models is highest score"
    difficulties = {pid: stats['failures'] for pid, stats in problem_stats.items()}
    sorted_difficulties = sorted(difficulties.items(), key=lambda x: x[1], reverse=True)
    
    sorted_ids_diff = [x[0] for x in sorted_difficulties]
    sorted_failures = [x[1] for x in sorted_difficulties]

    # Handle excessive number of IDs for plotting
    MAX_BARS = 60
    plot_ids_diff = sorted_ids_diff[:MAX_BARS]
    plot_failures_diff = sorted_failures[:MAX_BARS]

    plt.figure(figsize=(15, 7))
    plt.bar(plot_ids_diff, plot_failures_diff, color='salmon')
    plt.title("Top Problem IDs by Difficulty (Most Failures Across Models)")
    plt.xlabel("Problem ID")
    plt.ylabel("Total Failures")
    plt.xticks(rotation=90)
    plt.tight_layout()
    diff_plot_path = os.path.join(args.out_dir, "baseline_difficulty.png")
    plt.savefig(diff_plot_path)
    plt.close()
    print(f"Saved difficulty plot to {diff_plot_path}")

    # 2. Sort by average output length
    avg_lengths = {}
    for pid, stats in problem_stats.items():
        if stats['lengths']:
            avg_lengths[pid] = np.mean(stats['lengths'])
        else:
            avg_lengths[pid] = 0
            
    sorted_avg_lengths = sorted(avg_lengths.items(), key=lambda x: x[1], reverse=True)
    
    # Filter out empty lengths
    sorted_avg_lengths = [x for x in sorted_avg_lengths if x[1] > 0]
    
    sorted_ids_len = [x[0] for x in sorted_avg_lengths][:MAX_BARS]
    sorted_lengths = [x[1] for x in sorted_avg_lengths][:MAX_BARS]

    if sorted_ids_len:
        plt.figure(figsize=(15, 7))
        plt.bar(sorted_ids_len, sorted_lengths, color='skyblue')
        plt.title("Top Problem IDs by Average Output Length (Tokens)")
        plt.xlabel("Problem ID")
        plt.ylabel("Average Response Length (Tokens)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        len_plot_path = os.path.join(args.out_dir, "baseline_average_length.png")
        plt.savefig(len_plot_path)
        plt.close()
        print(f"Saved average length plot to {len_plot_path}")
    else:
        print("No length data available to plot.")

    # 3. Correlation between Difficulty and Average Length
    valid_pids = [pid for pid in avg_lengths if avg_lengths[pid] > 0]
    
    if valid_pids:
        x_diff = [difficulties[pid] for pid in valid_pids]
        y_len = [avg_lengths[pid] for pid in valid_pids]
        
        # Calculate correlation
        correlation = np.corrcoef(x_diff, y_len)[0, 1]
        print(f"Pearson Correlation between Difficulty and Average Length: {correlation:.4f}")
        
        plt.figure(figsize=(10, 7))
        plt.scatter(x_diff, y_len, alpha=0.6, color='purple')
        
        # Add a trendline
        z = np.polyfit(x_diff, y_len, 1)
        p = np.poly1d(z)
        plt.plot(x_diff, p(x_diff), "r--", alpha=0.8, label=f"Trendline (r={correlation:.2f})")
        
        plt.title("Problem Difficulty vs Average Response Length")
        plt.xlabel("Difficulty (Total Failures across Models)")
        plt.ylabel("Average Response Length (Tokens)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        corr_plot_path = os.path.join(args.out_dir, "baseline_difficulty_vs_length.png")
        plt.savefig(corr_plot_path)
        plt.close()
        print(f"Saved correlation plot to {corr_plot_path}")

if __name__ == "__main__":
    main()
