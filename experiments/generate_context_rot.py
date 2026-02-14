import argparse
import os
import sys
import json
import time
from vllm import LLM, SamplingParams
from tqdm import tqdm
import requests
from transformers import AutoTokenizer

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

def load_and_chunk_text(filepath, min_length=200):
    """
    Loads text from filepath, splits into paragraphs, and filters them.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

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


def load_and_chunk_text_by_tokens(filepath, target_tokens=3000, tokenizer=None):
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
             return load_and_chunk_text_approx(filepath, target_tokens)

    paragraphs = load_and_chunk_text(filepath, min_length=500)
    
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
    needed_turns = (args.target_tokens - total_tokens) // estimated_tokens_per_turn
    if needed_turns < 0: needed_turns = 1
    
    # Needed rounds of 20-distractors
    # Each round of 20 gives 5 turns (if distractors_per_query=4)
    distractors_per_query = args.distractors_per_query
    if args.distractor_type == "text":
        distractors_per_query = 1 # Force 1 chunk per query for text
        print("Using 1 text chunk (approx 1500 tokens) per query.")
        
    turns_per_round = 20 // distractors_per_query
    needed_rounds = (needed_turns // turns_per_round) + 2 # +2 buffer
    
    print(f"Targeting {args.target_tokens} tokens.")
    print(f"Estimated {needed_turns} turns needed.")
    print(f"Generating {needed_rounds} rounds of 20 distractors (Total {needed_rounds * 20} systems).")
    
    all_prompts = []
    
    current_system_index = 1
    
    # Pre-load text if needed
    text_chunks = []
    if args.distractor_type == "text":
        print(f"Ensuring text is available at {args.text_file}...")
        ensure_downloaded(args.text_url, args.text_file)
        # Using 1500 tokens per chunk
        text_chunks = load_and_chunk_text_by_tokens(args.text_file, target_tokens=2000, tokenizer=tokenizer)
        if not text_chunks:
            print("Error: No chunks loaded from text file.")
            return

    # 1. Pre-generate ALL prompts
    print("Preparing all prompts...")
    for r in tqdm(range(needed_rounds)):
        seed = int(time.time() * 1000) + r
        
        distractor_pool = []
        if args.distractor_type == "math":
            # Generate 20 distractors with correct absolute definition indexing
            distractor_pool = generate_20_distractors(lcase_dict, ucase_dict, greek_dict, seed, start_index=current_system_index)
        else:
            # Text distractors: slice 20 chunks, wrapping around
            base_idx = (r * 20) % len(text_chunks)
            for k in range(20):
                p_idx = (base_idx + k) % len(text_chunks)
                distractor_pool.append(text_chunks[p_idx])
        
        # Chunk into groups (e.g. 4 for math, 1 for text)
        chunk_size = distractors_per_query
        for i in range(0, len(distractor_pool), chunk_size):
            chunk = distractor_pool[i : i + chunk_size]
            if not chunk: continue
            
            # Calculate global start/end indices for this chunk
            start_num = current_system_index + i
            end_num = start_num + len(chunk) - 1
            
            # Create prompt for this chunk
            if args.distractor_type == "math":
                prompt_text = f"Here are {len(chunk)} mathematical systems. Analyze each and answer the verification question for each. Number your answers {start_num} to {end_num}.\n\n"
                for j, s in enumerate(chunk):
                     prompt_text += f"{start_num + j}. {s}\n\n"
                prompt_text += "Answer:\n"
            else:
                # Text: single chunk
                prompt_text = f"Here is a text excerpt. Analyze the main argument, rhetorical style and historical context of this text. \n\nText {start_num}:\n{chunk[0]}\n\nAnswer {start_num}:\n"
            
            all_prompts.append(prompt_text)
           
        current_system_index += 20

    print(f"Prepared {len(all_prompts)} prompts. Starting generation...")
    
    # 2. Generate ALL in parallel
    # vLLM will batch this efficiently across GPUs
    start_gen = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    gen_time = time.time() - start_gen
    print(f"Generation took {gen_time:.2f}s")
    
    # 3. Serialize Results
    print("Processing outputs and saving...")
    new_tokens = 0
    pbar_gen = tqdm(total=len(outputs), unit="turn")
    
    for output in outputs:
        generated = output.outputs[0].text
        prompt_text = output.prompt
        
        # Calculate tokens
        prompt_ids = output.prompt_token_ids
        output_ids = output.outputs[0].token_ids
        count = len(prompt_ids) + len(output_ids)
        new_tokens += count
        total_tokens += count
        
        history.append({
            "role": "user",
            "content": prompt_text
        })
        history.append({
            "role": "assistant",
            "content": generated
        })
        pbar_gen.update(1)
        
        if total_tokens >= args.target_tokens:
             print(f"Reached target tokens: {total_tokens}")
             break
             
    # Save once at the end
    with open(args.output_file, "w") as f:
        json.dump(history, f, indent=2)
        
    pbar_gen.close()

    elapsed = time.time() - start_time
    print(f"Finished! Generated {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.2f} tok/s).")
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    main()
