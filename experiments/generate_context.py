import argparse
import os
import sys
import json
import time
from vllm import LLM, SamplingParams
from tqdm import tqdm
import requests
from transformers import AutoTokenizer
import numpy as np

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.context_saturation.generate_systems_static import generate_20_distractors, lcase_dict, ucase_dict, greek_dict


# Mock vLLM for verification
class MockSamplingParams:
    def __init__(self, temperature=0.7, max_tokens=2048):
        pass

class MockLLM:
    def __init__(self, model, tensor_parallel_size=1, max_model_len=4096):
        pass
    
    def get_tokenizer(self):
        return None
        
    def generate(self, prompts, sampling_params):
        results = []
        for p in prompts:
            # Create a mock output object structure that matches vLLM
            class MockOutput:
                def __init__(self, prompt):
                    self.prompt = prompt
                    self.prompt_token_ids = [1] * (len(prompt) // 4) # Fake token count
                    
                    class OutputItem:
                        def __init__(self):
                            self.text = "Mock Answer"
                            self.token_ids = [2] * 50
                    self.outputs = [OutputItem()]
                    
            results.append(MockOutput(p))
        return results


def ensure_downloaded(url, filepath):
    """
    Checks if a file exists at filepath. If not, downloads it from url.
    """
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return

    print(f"Downloading from {url} to {filepath}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        # Clean up partial file if needed
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

def load_and_chunk_text(filepath, min_length=200, start_marker=None):
    """
    Loads text from filepath, splits into paragraphs, and filters them.
    If start_marker is provided, only text starting from that marker is processed.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
        
    if start_marker:
        start_idx = text.find(start_marker)
        if start_idx != -1:
            print(f"Found start marker '{start_marker}' at index {start_idx}. Trimming previous text.")
            text = text[start_idx:]
        else:
            print(f"Warning: Start marker '{start_marker}' not found in {filepath}. Using full text.")

    # Split by double newlines (common for Project Gutenberg texts)
    # Also handle common Gutenberg headers/footers roughly if possible, 
    # but for now we just filter by length.
    paragraphs = text.split('\n\n')
    
    clean_paragraphs = []
    for p in paragraphs:
        # Replace single newlines within a paragraph with spaces
        clean_p = p.replace('\n', ' ').strip()
        
        if len(clean_p) >= min_length:
            clean_paragraphs.append(clean_p)
            
    print(f"Loaded {len(clean_paragraphs)} paragraphs from {filepath}.")
    return clean_paragraphs


def load_and_chunk_text_by_tokens(filepath, target_tokens=3000, tokenizer=None, start_marker=None):
    """
    Loads text, splits into paragraphs, and then groups them into chunks 
    that are approximately target_tokens long.
    Uses accurate token counting via tokenizer.
    """
    if tokenizer is None:
        print("No tokenizer provided for chunking, loading gpt2 default...")
        try:
             tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
             print(f"Failed to load gpt2 tokenizer: {e}. Falling back to approximation.")
             # Fallback to approx if needed, but user requested real tokenizer.
             # Note: load_and_chunk_text_approx isn't updated for start_marker here, assume tokenizer works usually.
             return load_and_chunk_text_approx(filepath, target_tokens)

    paragraphs = load_and_chunk_text(filepath, min_length=500, start_marker=start_marker)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Pre-calculate to avoid repeated encoding? 
    # Or just loop. Tokenizing single paragraphs is fast.
    print(f"Tokenizing {len(paragraphs)} paragraphs for accurate chunking...")

    for p in paragraphs:
        # Encode
        token_ids = tokenizer.encode(p, add_special_tokens=False)
        p_len = len(token_ids)
        
        # If adding this paragraph exceeds target significantly, finish current chunk
        # But if current chunk is empty, take it anyway (unless it's massive, but we assume paragraphs are reasonable)
        if current_chunk and (current_length + p_len > target_tokens):
             # Check if we are closer to target with or without this paragraph?
             # Simple logic: once we cross target, or are close enough, stop.
             # Here we just fill until > target
             chunks.append("\n\n".join(current_chunk))
             current_chunk = [p]
             current_length = p_len
        else:
             current_chunk.append(p)
             current_length += p_len
             
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    print(f"Created {len(chunks)} chunks of approx {target_tokens} tokens from {filepath}.")
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Generate Context Rot (Distractors + Answers)")
    parser.add_argument("--model_path", type=str, default="tiiuae/Falcon-H1R-7B", help="Model to use for generation")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--target_tokens", type=int, default=1000000, help="Total tokens (User+Assistant) to generate")
    parser.add_argument("--distractors_per_query", type=int, default=4, help="Number of distractors per user prompt")
    parser.add_argument("--num_gpu", type=int, default=1, help="TP size for vLLM")
    parser.add_argument("--mock", action="store_true", help="Use mock vLLM for verification")
    parser.add_argument("--distractor_type", type=str, default="math", choices=["math", "text"], help="Type of distractors to generate")
    parser.add_argument("--text_url", type=str, default="https://www.gutenberg.org/cache/epub/25717/pg25717.txt", help="URL for text source")
    parser.add_argument("--text_file", type=str, default="experiments/data/gibbon_vol1.txt", help="Local path for text source")
    parser.add_argument("--start_marker", type=str, default="In the second century of the Christian", help="String to mark start of text processing")
    
    args = parser.parse_args()
    start_time = time.time()

    print(f"Initializing vLLM with model: {args.model_path}")
    
    llm = None
    tokenizer = None
    
    if args.mock:
        print("MOCK MODE: Using mock vLLM.")
        # Load tokenizer for chunking if possible
        try:
             print(f"Loading tokenizer from {args.model_path} for chunking...")
             tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        except Exception as e:
             print(f"Warning: Could not load tokenizer from {args.model_path}: {e}. using gpt2.")
             tokenizer = AutoTokenizer.from_pretrained("gpt2")
             
        llm = MockLLM(model=args.model_path, tensor_parallel_size=args.num_gpu, max_model_len=8192)
    else:
        # from vllm import LLM, SamplingParams
        llm = LLM(model=args.model_path, tensor_parallel_size=args.num_gpu, max_model_len=8192)
        tokenizer = llm.get_tokenizer()
    
    # Sampling parameters for producing varied but reasonable answers
    sampling_params = SamplingParams( temperature=0.7, max_tokens=8192)

    history = []
    total_tokens = 0
    
    # Check if file exists to resume (load existing history)
    if os.path.exists(args.output_file):
        print(f"Loading existing history from: {args.output_file}")
        try:
            with open(args.output_file, 'r') as f:
                history = json.load(f)
            print(f"Loaded {len(history)} turns.")
        except Exception as e:
            print(f"Error loading existing file: {e}. Starting fresh.")
            history = []
    
    pbar = tqdm(total=args.target_tokens, unit="tok", initial=total_tokens)

    estimated_tokens_per_turn = 9200
    if args.distractor_type == "text":
         # Text usually has ~2000 prompt + ~3000 completion = ~5000. 
         # Use a conservative estimate to generate enough turns.
         estimated_tokens_per_turn = 5000 
         
    needed_turns = (args.target_tokens - total_tokens) // estimated_tokens_per_turn
    if needed_turns < 0: needed_turns = 1
    
    # Needed rounds of 20-distractors
    # Each round of 20 gives 5 turns (if distractors_per_query=4)
    # Calculate how many distractors per query
    distractors_per_query = args.distractors_per_query
    if args.distractor_type == "text":
        distractors_per_query = 1 # Force 1 chunk per query for text
        print("Using 1 text chunk (approx 1500-2000 tokens) per query.")
        
    print(f"Targeting at least {args.target_tokens} tokens.")
    
    current_system_index = 1
    
    # Pre-load text if needed
    text_chunks = []
    if args.distractor_type == "text":
        print(f"Ensuring text is available at {args.text_file}...")
        ensure_downloaded(args.text_url, args.text_file)
        # Using 2000 tokens per chunk target
        text_chunks = load_and_chunk_text_by_tokens(args.text_file, target_tokens=2000, tokenizer=tokenizer, start_marker=args.start_marker)
        if not text_chunks:
            print("Error: No chunks loaded from text file.")
            return

    # Dynamic Generation Loop
    print("Starting dynamic generation loop...")
    
    # Use a fixed batch size (e.g. 100 distractors = 25 prompts for math, 100 for text)
    # A "round" of 20 was used before, let's stick to batches of ~20-50 distractors
    batch_size_distractors = 50 
    
    # Keep track of batches
    batch_count = 0
    
    # Save incrementally
    # We will accumulate history in memory but write to file periodically or at end?
    # Actually, let's keep history in memory for analysis at end, but maybe write to file if huge?
    # For now, keep simple: append to history, check total_tokens.
    
    # If resuming, total_tokens is already set
    
    while total_tokens < args.target_tokens:
        batch_count += 1
        batch_prompts = []
        
        # Generate a batch of distractors
        distractor_pool = []
        
        # How many chunks do we need for this batch?
        # We want `batch_size_distractors` items in the pool
        
        if args.distractor_type == "math":
            # Generate batch_size_distractors distractors
            # Seed based on time + batch index
            seed = int(time.time() * 1000) + batch_count
            distractor_pool = generate_20_distractors(lcase_dict, ucase_dict, greek_dict, seed, start_index=current_system_index, count=batch_size_distractors)
            # generate_20_distractors usually returns 20. modifying it to return flexible count or calling it multiple times?
            # It seems hardcoded to 20 ("generate_20_distractors"). 
            # Let's just call it multiple times if needed, or just work with chunks of 20.
            # To be safe and consistent with previous logic:
            pass
        
        # Actually, let's just do "Rounds" of 20 inside the loop until we have enough for a batch, 
        # OR just do 1 round of 20 per loop iteration. Simpler.
        
        # 1 Round of 20 distractors
        seed = int(time.time() * 1000) + batch_count
        
        distractor_pool = []
        if args.distractor_type == "math":
             distractor_pool = generate_20_distractors(lcase_dict, ucase_dict, greek_dict, seed, start_index=current_system_index)
        else:
             # Text: slice 20 chunks
             # wrap around text chunks
             base_idx = ((batch_count - 1) * 20) % len(text_chunks)
             for k in range(20):
                 p_idx = (base_idx + k) % len(text_chunks)
                 distractor_pool.append(text_chunks[p_idx])
                 
        # Create prompts from pool
        for i in range(0, len(distractor_pool), distractors_per_query):
            chunk = distractor_pool[i : i + distractors_per_query]
            if not chunk: continue
            
            start_num = current_system_index + i
            end_num = start_num + len(chunk) - 1
            
            prompt_text = ""
            if args.distractor_type == "math":
                prompt_text = f"Here are {len(chunk)} mathematical systems. Analyze each and answer the verification question for each. Number your answers {start_num} to {end_num}.\n\n"
                for j, s in enumerate(chunk):
                     prompt_text += f"{start_num + j}. {s}\n\n"
                prompt_text += "Answer:\n"
            else:
                prompt_text = f"Here is a text excerpt. Analyze the main argument, rhetorical style and historical context of this text. \n\nText {start_num}:\n{chunk[0]}\n\nAnswer {start_num}:\n"
            
            batch_prompts.append(prompt_text)

        current_system_index += 20
        
        # Run inference for this batch
        if not batch_prompts: continue
        
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process outputs
        for output in outputs:
            generated = output.outputs[0].text
            prompt_text = output.prompt
            
            # Calculate tokens
            prompt_ids = output.prompt_token_ids
            output_ids = output.outputs[0].token_ids
            count = len(prompt_ids) + len(output_ids)
            
            history.append({
                "role": "user",
                "content": prompt_text
            })
            history.append({
                "role": "assistant",
                "content": generated
            })
            
            total_tokens += count
            pbar.update(count)
            
            if total_tokens >= args.target_tokens:
                 print(f"Reached target tokens: {total_tokens}")
                 break
        
        if total_tokens >= args.target_tokens:
            break
            
    pbar.close()
    


    # Truncate to exact target tokens
    if tokenizer and total_tokens > args.target_tokens:
        print(f"Trimming output to exact target tokens: {args.target_tokens}...")
        current_count = 0
        trimmed_history = []
        
        for entry in history:
            content = entry["content"]
            token_ids = tokenizer.encode(content) # Standard encoding
            count = len(token_ids)
            
            if current_count + count <= args.target_tokens:
                trimmed_history.append(entry)
                current_count += count
            else:
                remaining = args.target_tokens - current_count
                if remaining > 0:
                    # Truncate this entry
                    # Decode only the remaining tokens
                    keep_ids = token_ids[:remaining]
                    new_content = tokenizer.decode(keep_ids, skip_special_tokens=True)
                    entry["content"] = new_content
                    trimmed_history.append(entry)
                    current_count += remaining
                break
        
        history = trimmed_history
        total_tokens = current_count
        print(f"Trimmed successfully. Final count: {total_tokens}")

    # Save final results (corrected)
    print("Saving results...")
    with open(args.output_file, "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    


    elapsed = time.time() - start_time
    print(f"Finished! Generated {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.2f} tok/s).")
    print(f"Saved to: {args.output_file}")
    
    # Run analysis
    if tokenizer:
        analyze_token_usage(history, tokenizer)


def analyze_token_usage(history, tokenizer):
    """
    Analyzes token usage in the generated history using the provided tokenizer.
    Adapted from analyze_tokens.py.
    """
    print("Analyzing token usage...")
    
    prompt_lengths = []
    completion_lengths = []
    total_lengths = []

    current_prompt_len = 0
    
    for entry in history:
        role = entry.get("role")
        content = entry.get("content", "")
        # Use simple whitespace split if tokenizer fails or is None (though it shouldn't be)
        if tokenizer:
            tokens = len(tokenizer.encode(content))
        else:
             tokens = len(content.split()) # Very rough fallback
        
        if role == "user":
            current_prompt_len = tokens
        elif role == "assistant":
            if current_prompt_len > 0:
                prompt_lengths.append(current_prompt_len)
                completion_lengths.append(tokens)
                total_lengths.append(current_prompt_len + tokens)
            current_prompt_len = 0 # Reset

    if not total_lengths:
        print("No valid user-assistant pairs found for analysis.")
        return

    print(f"  Total Queries: {len(total_lengths)}")
    print(f"  Prompt Tokens:    Mean={np.mean(prompt_lengths):.1f}, Median={np.median(prompt_lengths):.1f}, Min={np.min(prompt_lengths)}, Max={np.max(prompt_lengths)}")
    print(f"  Completion Tokens: Mean={np.mean(completion_lengths):.1f}, Median={np.median(completion_lengths):.1f}, Min={np.min(completion_lengths)}, Max={np.max(completion_lengths)}")
    print(f"  Total per Query:  Mean={np.mean(total_lengths):.1f}, Median={np.median(total_lengths):.1f}, Min={np.min(total_lengths)}, Max={np.max(total_lengths)}")
    print("-" * 40)


if __name__ == "__main__":
    main()
