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
    from experiments.context_saturation.generate_systems_static import generate_systems_static
except ImportError:
    # Fallback if running from a different directory
    from context_saturation.generate_systems_static import generate_systems_static

def main():
    parser = argparse.ArgumentParser(description="Generate Context Rot (Distractors + Answers)")
    parser.add_argument("--model_path", type=str, default="tiiuae/Falcon-H1R-7B", help="Model to use for generation")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--target_tokens", type=int, default=5000000, help="Total tokens (User+Assistant) to generate")
    parser.add_argument("--distractors_per_query", type=int, default=8, help="Number of distractors per user prompt")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of concurrent queries to run")
    parser.add_argument("--num_gpu", type=int, default=1, help="TP size for vLLM")
    
    args = parser.parse_args()

    print(f"Initializing vLLM with model: {args.model_path}")
    llm = LLM(model=args.model_path, tensor_parallel_size=args.num_gpu, max_model_len=4096) # Limit context for generation speed
    tokenizer = llm.get_tokenizer()
    
    # Sampling parameters for producing varied but reasonable answers
    sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=2048, # Enough for 8 answers? 8 * 200 = 1600.
    )

    history = []
    
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
    
    pbar = tqdm(total=args.target_tokens, unit="tok")

    start_time = time.time()
    
    while total_tokens < args.target_tokens:
        # 1. Prepare Batch of Prompts (Sequential for history)
        # We generate ONE turn at a time to maintain the linear history structure requested
        # "Here are 8 systems... Answer: [1..8]"
       
        prompts = []
        raw_distractor_groups = []
        
        # terms_pool = ["Alice", "Bob", "n", "stack", "x", "y", "z", "u", "v", "w", "alpha", "beta"]
        
        for _ in range(args.batch_size):
            sys_list = generate_systems_static(args.distractors_per_query)
            
            # Format Prompt matching User Request somewhat
            # User example: "Let's define System 1..."
            # Our generate_systems returns "Let us define System-X..." strings.
            prompt_text = f"Here are {args.distractors_per_query} mathematical systems. Analyze each and answer the verification question for each. Number your answers 1 to {args.distractors_per_query}.\n\n"
            for i, s in enumerate(sys_list):
                 prompt_text += f"{i+1}. {s}\n\n"
            
            prompt_text += "Answer:\n"
            prompts.append(prompt_text)

        # 2. Generate
        outputs = llm.generate(prompts, sampling_params)

        # 3. Process and Append to History
        batch_tokens = 0
        
        for output in outputs:
            prompt = output.prompt
            generated = output.outputs[0].text
            
            # Calculate tokens (approximated or precise)
            prompt_ids = output.prompt_token_ids
            output_ids = output.outputs[0].token_ids
            count = len(prompt_ids) + len(output_ids)
            batch_tokens += count
            
            # Append as User/Assistant pair
            history.append({
                "role": "user",
                "content": prompt
            })
            history.append({
                "role": "assistant",
                "content": generated
            })

        total_tokens += batch_tokens
        pbar.update(batch_tokens)

        # Save Incrementally (Overwrite file with full JSON list)
        # efficient enough for 20MB file
        with open(args.output_file, "w") as f:
            json.dump(history, f, indent=2)

    pbar.close()

    pbar.close()
    elapsed = time.time() - start_time
    print(f"Finished! Generated {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.2f} tok/s).")
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    main()
