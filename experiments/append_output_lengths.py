import os
import json
import glob
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

TOKENIZERS = {}

def get_exact_tokenizer(model_dir):
    if model_dir in TOKENIZERS:
        return TOKENIZERS[model_dir]

    print(f"\nLoading exact tokenizer for {model_dir}...")

    # 1. OpenAI
    if model_dir in ["gpt-5.4", "openai_gpt-oss-120b"]:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        func = lambda text: len(enc.encode(str(text)))
        TOKENIZERS[model_dir] = func
        return func

    # 2. Known local HF models
    hf_id = None
    if model_dir == "nvidia_OpenReasoning-Nemotron-7B":
        hf_id = "nvidia/OpenReasoning-Nemotron-7B"
    elif model_dir == "nvidia_OpenReasoning-Nemotron-32B":
        hf_id = "nvidia/OpenReasoning-Nemotron-32B"
    elif model_dir == "Qwen_Qwen3-30B-A3B-Thinking-2507":
        hf_id = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    elif model_dir == "deepseek-ai_DeepSeek-R1-Distill-Llama-70B":
        hf_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    elif model_dir == "tiiuae_Falcon-H1R-7B":
        hf_id = "tiiuae/Falcon-H1R-7B"
    elif model_dir == "GAIR_LIMO-v2":
        hf_id = "GAIR/LIMO-v2"

    if hf_id:
        try:
            from transformers import AutoTokenizer
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            func = lambda text: len(tok.encode(str(text), add_special_tokens=False))
            TOKENIZERS[model_dir] = func
            return func
        except Exception as e:
            raise ValueError(f"CRITICAL: Failed to load exact HF tokenizer '{hf_id}' for '{model_dir}': {e}")

    # 3. Known API models (Google / Anthropic)
    if 'gemini' in model_dir.lower():
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(f"CRITICAL: GOOGLE_API_KEY not set for {model_dir}")
        client = genai.Client(api_key=api_key)
        
        def google_counter(text):
            return client.models.count_tokens(
                model=model_dir,
                contents=str(text),
            ).total_tokens
            
        TOKENIZERS[model_dir] = google_counter
        return google_counter
        
    elif 'claude' in model_dir.lower():
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(f"CRITICAL: ANTHROPIC_API_KEY not set for {model_dir}")
        client = Anthropic(api_key=api_key, default_headers={"anthropic-beta": "context-1m-2025-08-07"})
        
        def anthropic_counter(text):
            resp = client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                model=model_dir,
                messages=[{"role": "user", "content": str(text)}]
            )
            # Subtracting approximate message formatting tokens if desired, 
            # but this is the strictest possible API token count available.
            return resp.input_tokens
            
        TOKENIZERS[model_dir] = anthropic_counter
        return anthropic_counter

    # Catch-all
    raise ValueError(f"CRITICAL: Could not find exact tokenizer mapping for '{model_dir}'.")

def append_to_file(filepath):
    # Only process files under results
    parts = filepath.split(os.sep)
    if "results" not in parts:
        return
        
    res_idx = parts.index("results")
    if len(parts) <= res_idx + 1:
        return
    model_name = parts[res_idx + 1]

    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            return

    modified = False
    tokenizer = None

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        has_summary_wrapper = True
    elif isinstance(data, list):
        results = data
        has_summary_wrapper = False
    else:
        return

    # Phase 1: Count individual records
    total_tokens = 0
    calculated_count = 0
    already_counted = 0
    
    for r in results:
        # Ignore summary dictionaries inside arrays
        if isinstance(r, dict) and r.get("id") is None and "summary" in r:
            continue
            
        if not isinstance(r, dict):
            continue
            
        text = r.get("output", "")
        if text is None:
            text = ""

        if "output_tokens" in r:
            # Check length is valid
            total_tokens += r["output_tokens"]
            already_counted += 1
            continue

        # If not present, we must invoke the tokenizer
        if tokenizer is None:
            tokenizer = get_exact_tokenizer(model_name)
            
        tok_len = tokenizer(text)
        r["output_tokens"] = tok_len
        total_tokens += tok_len
        calculated_count += 1
        modified = True

    valid_count = calculated_count + already_counted
    
    # Phase 2: Update Average in Summary 
    if valid_count > 0:
        avg_tokens = total_tokens / valid_count
        
        if has_summary_wrapper:
            if "summary" not in data:
                data["summary"] = {}
                
            current_avg = data["summary"].get("avg_output_tokens", None)
            if current_avg != avg_tokens:
                data["summary"]["avg_output_tokens"] = avg_tokens
                modified = True
        else:
            summary_found = False
            if len(results) > 0 and isinstance(results[-1], dict) and "summary" in results[-1]:
                summary_block = results[-1]["summary"]
                current_avg = summary_block.get("avg_output_tokens", None)
                if current_avg != avg_tokens:
                    summary_block["avg_output_tokens"] = avg_tokens
                    modified = True
                summary_found = True
                
            if not summary_found:
                new_summary = {"avg_output_tokens": avg_tokens}
                results.append({"summary": new_summary})
                modified = True

    if modified:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Append output lengths to all JSON result files.")
    parser.add_argument("--experiments_dir", type=str, default="experiments", help="Base directory")
    args = parser.parse_args()

    # Find JSONs recursively under the provided directory
    pattern = os.path.join(args.experiments_dir, "**", "*.json")
    all_files = glob.glob(pattern, recursive=True)
    
    files_to_process = []
    for f in all_files:
        basename = os.path.basename(f)
        if basename.startswith("jobs_") or basename.startswith("tracking_") or basename.startswith("batch_"):
            continue
        if "_summary_" in basename or "prompt_recovery" in basename:
            continue
        if "/analysis/" in f:
            continue
        files_to_process.append(f)

    print(f"Found {len(files_to_process)} result files. Checking for missing lengths...")
    
    failure = False
    for filepath in tqdm(files_to_process):
        try:
            append_to_file(filepath)
        except Exception as e:
            if "CRITICAL" in str(e):
                print(f"\n{e}\nFile: {filepath}")
                failure = True
                break
            else:
                print(f"\nError in {filepath}: {e}")

    if failure:
        print("\nProcess halted due to precise tokenizer requirement failure.")
    else:
        print("\nFinished successfully processing all result files.")

if __name__ == "__main__":
    main()
