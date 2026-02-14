import argparse
import os
import sys
import json
import time
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from experiments.context_saturation.generate_systems_static import generate_20_distractors, lcase_dict, ucase_dict, greek_dict
except ImportError:
    # Fallback if running from a different directory
    from context_saturation.generate_systems_static import generate_20_distractors, lcase_dict, ucase_dict, greek_dict

def main():
    parser = argparse.ArgumentParser(description="Generate Context Rot (Distractors + Answers)")
    parser.add_argument("--model_path", type=str, default="tiiuae/Falcon-H1R-7B", help="Model to use for generation")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--target_tokens", type=int, default=1000000, help="Total tokens (User+Assistant) to generate")
    parser.add_argument("--distractors_per_query", type=int, default=8, help="Number of distractors per user prompt")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of concurrent queries to run")
    parser.add_argument("--num_gpu", type=int, default=1, help="TP size for vLLM")
    parser.add_argument("--mock", action="store_true", help="Use mock vLLM for verification")
    
    args = parser.parse_args()
    start_time = time.time()

    print(f"Initializing vLLM with model: {args.model_path}")
    
    if args.mock:
        print("MOCK MODE: Using mock vLLM.")
        from experiments.mock_vllm import LLM, SamplingParams
        llm = LLM(model=args.model_path, tensor_parallel_size=args.num_gpu, max_model_len=4096)
        tokenizer = None
    else:
        llm = LLM(model=args.model_path, tensor_parallel_size=args.num_gpu, max_model_len=4096) # Limit context for generation speed
        tokenizer = llm.get_tokenizer()
    
    # Sampling parameters for producing varied but reasonable answers
    sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=2048, # Enough for 8 answers? 8 * 200 = 1600.
    )

    history = []
    total_tokens = 0
    
    # Check if file exists to resume (load existing history)
    if os.path.exists(args.output_file):
        print(f"Loading existing history from: {args.output_file}")
        try:
            with open(args.output_file, 'r') as f:
                history = json.load(f)
            # Recalculate token count roughly (or assume it's part of the job)
            # For simplicity, we just append newly generated tokens to the target count
            # but we won't re-verify the old count deeply to avoid slow startup.
            print(f"Loaded {len(history)} turns.")
        except Exception as e:
            print(f"Error loading existing file: {e}. Starting fresh.")
            history = []
    
    pbar = tqdm(total=args.target_tokens, unit="tok", initial=total_tokens)

    # Calculate how many total systems/prompts we need
    # Target: 1M tokens. 
    # Approximation: 
    # Each distractor is ~200-300 tokens? 
    # A set of 4 distractors + answer is around 9200 tokens?
    # Let's say 9200 tokens per "turn" (User + Assistant).
    # 1,000,000 / 9200 = ~108 turns.
    # We can refine this or just generate a large buffer. 
    # Better strategy: Generate rounds of 20 distractors (5 turns of 4) until we likely exceed target.
    
    estimated_tokens_per_turn = 9200
    needed_turns = (args.target_tokens - total_tokens) // estimated_tokens_per_turn
    if needed_turns < 0: needed_turns = 1
    
    # Needed rounds of 20-distractors
    # Each round of 20 gives 5 turns (if distractors_per_query=4)
    turns_per_round = 20 // args.distractors_per_query
    needed_rounds = (needed_turns // turns_per_round) + 2 # +2 buffer
    
    print(f"Targeting {args.target_tokens} tokens.")
    print(f"Estimated {needed_turns} turns needed.")
    print(f"Generating {needed_rounds} rounds of 20 distractors (Total {needed_rounds * 20} systems).")
    
    all_prompts = []
    
    current_system_index = 1
    
    # 1. Pre-generate ALL prompts
    print("Preparing all prompts...")
    for r in tqdm(range(needed_rounds)):
        seed = int(time.time() * 1000) + r
        # Generate 20 distractors with correct absolute definition indexing
        # Note: generate_20_distractors now accepts start_index
        distractor_pool = generate_20_distractors(lcase_dict, ucase_dict, greek_dict, seed, start_index=current_system_index)
        
        # Chunk into groups (e.g. 4)
        chunk_size = args.distractors_per_query
        for i in range(0, len(distractor_pool), chunk_size):
            chunk = distractor_pool[i : i + chunk_size]
            if not chunk: continue
            
            # Calculate global start/end indices for this chunk
            start_num = current_system_index + i
            end_num = start_num + len(chunk) - 1
            
            # Create prompt for this chunk
            prompt_text = f"Here are {len(chunk)} mathematical systems. Analyze each and answer the verification question for each. Number your answers {start_num} to {end_num}.\n\n"
            for j, s in enumerate(chunk):
                 prompt_text += f"{start_num + j}. {s}\n\n"
            prompt_text += "Answer:\n"
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
