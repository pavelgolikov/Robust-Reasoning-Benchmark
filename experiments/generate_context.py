import argparse
import os
import sys
import json
import time
import random
from tqdm import tqdm

# Ensure local imports work whether run from root or experiments/
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from util import BASELINE_SYSTEM_PROMPT
from context_saturation.generate_systems_static import generate_20_distractors
from context_saturation.generate_systems_static import lcase_dict, ucase_dict, greek_dict

def main():
    parser = argparse.ArgumentParser(description="Generate context rot (distractors) using vLLM")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Model name/path")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the JSON history")
    parser.add_argument("--token_target", type=int, default=100000, help="Stop when context exceeds this tokens")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of distractors to generate in parallel")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs for vLLM")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Max model length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry", action="store_true", help="Dry run without loading model")
    args = parser.parse_args()
    random.seed(args.seed)
    
    # 1. Initialize vLLM
    llm = None
    tokenizer = None
    sampling_params = None
    
    if not args.dry:
        print(f"Initializing vLLM with model: {args.model}")
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.num_gpus,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            dtype="bfloat16"
        )
        tokenizer = llm.get_tokenizer()
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_model_len)
    else:
        print("Dry run: Skipping model load.")
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2") # Fallback
        except:
            tokenizer = None

    # 2. Main Generation Loop
    history = []
    
    history.append({
        "role": "system",
        "content": BASELINE_SYSTEM_PROMPT
    })
    
    current_token_count = 0
    if tokenizer:
        current_token_count = len(tokenizer.encode(BASELINE_SYSTEM_PROMPT))
    
    print(f"Target Tokens: {args.token_target}")
    
    batch_idx = 0
    
    # To use generate_20_distractors effectively, we need to manage seeds or indices
    # generate_20_distractors(..., seed, start_index)
    distractor_index = 1
    
    pbar = tqdm(total=args.token_target, desc="Generating Tokens", unit="tok")
    
    while current_token_count < args.token_target:
        # Generate enough distractors for the batch
        distractors_batch = []
        
        while len(distractors_batch) < args.batch_size:
            # Generate 20 at a time
            # We vary seed to get different permutations
            batch_seed = args.seed + batch_idx + distractor_index
            new_distractors = generate_20_distractors(lcase_dict, ucase_dict, greek_dict, batch_seed)
            
            # Add to batch
            distractors_batch.extend(new_distractors)
            distractor_index += 20
        
        # Trim to exact batch size if needed (though not strictly necessary)
        distractors_batch = distractors_batch[:args.batch_size]
        
        # Prepare Prompts for vLLM
        # Each prompt needs the system prompt + user question
        prompts = []
        for d in distractors_batch:
            # Format: System + User
            msgs = [
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": d}
            ]
            # We want the model to generate the assistant response
            full_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(full_prompt)
            
        # Generate
        if not args.dry:
            outputs = llm.generate(prompts, sampling_params)
            responses = [o.outputs[0].text for o in outputs]
        else:
            responses = [f"Mock Answer {i} \\boxed{{{i}}}" for i in range(len(prompts))]
            
        # Append to History and Count Tokens
        for d, r in zip(distractors_batch, responses):
            # User Message
            user_msg = {"role": "user", "content": d}
            history.append(user_msg)
            
            # Assistant Message
            asst_msg = {"role": "assistant", "content": r}
            history.append(asst_msg)
            
            # Update Token Count
            if tokenizer:
                # Approximate or Exact
                # Exact: encode the new content
                t_u = len(tokenizer.encode(d))
                t_a = len(tokenizer.encode(r))
                added = t_u + t_a
                current_token_count += added
                pbar.update(added)
            else:
                # Mock count
                added = len(d)//4 + len(r)//4
                current_token_count += added
                pbar.update(added)

        batch_idx += 1
        
        # Intermediate Save (optional but good for safety)
        if batch_idx % 5 == 0:
            with open(args.output_file, 'w') as f:
                json.dump(history, f, indent=2)

    pbar.close()
    
    # Final Save
    print(f"Finished. Total Tokens: {current_token_count}")
    
    # Create directory if needed
    out_dir = os.path.dirname(args.output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.output_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved context to {args.output_file}")

if __name__ == "__main__":
    main()
