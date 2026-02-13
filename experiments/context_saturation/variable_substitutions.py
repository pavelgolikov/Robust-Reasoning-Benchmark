import argparse
import os
import json
import time
from datasets import load_dataset
# vLLM imports will be done inside main or conditionally to allow lightweight help check
from vllm import LLM, SamplingParams

def get_substitution_prompt(problem_text, lower_vars, upper_vars, names):
    """
    Constructs the prompt for the model to rewrite the problem.
    """
    prompt = f"""You are a data augmentation assistant for a math dataset. 
Your task is to rewrite the given math problem by substituting its existing variable names and proper names with standardized ones from the provided lists.

GUIDELINES:
0. If the candidate variable already exists in the problem, do not replace it.
1. Replace existing lower-case variables (like x, y, z, alpha, etc.) with variables from the 'Lower Case Candidates' list.
2. Replace existing upper-case variables (like A, B, C, etc.) with variables from the 'Upper Case Candidates' list.
3. Replace existing proper names with names from the 'Name Candidates' list.
5. KEEP the mathematical structure, logic, and numbers EXACTLY the same. Only the identifiers should change.
6. Output ONLY the rewritten problem statement. Do not output any explanations.

CANDIDATE LISTS:
- Lower Case Candidates: {', '.join(lower_vars)}
- Upper Case Candidates: {', '.join(upper_vars)}
- Name Candidates: {', '.join(names)}

ORIGINAL PROBLEM:
{problem_text}

REWRITTEN PROBLEM:
"""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Rewrite dataset problems by substituting variables and names using an LLM.")
    
    # Inputs
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the modified dataset (JSON format)")
    
    # Model config
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Name/Path of the model to use for rewriting")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for vLLM")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Max model context length")
    
    # Substitution Lists
    parser.add_argument("--lower_vars", type=str, default="x,n,m,k", help="Comma-separated list of lower case variable candidates")
    parser.add_argument("--upper_vars", type=str, default="A,B,C,D", help="Comma-separated list of upper case variable candidates")
    parser.add_argument("--names", type=str, default="Alice,Bob,Carol,David", help="Comma-separated list of name candidates")
    
    # Processing
    parser.add_argument("--sample_range", type=str, default=None, help="Range of indices to process (e.g. '0-10' or '1,3,5'). If None, processes all.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Parse Lists
    lower_vars = [x.strip() for x in args.lower_vars.split(',') if x.strip()]
    upper_vars = [x.strip() for x in args.upper_vars.split(',') if x.strip()]
    names_list = [x.strip() for x in args.names.split(',') if x.strip()]
    
    print(f"Candidates:")
    print(f"  Lower: {lower_vars}")
    print(f"  Upper: {upper_vars}")
    print(f"  Names: {names_list}")

    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    try:
        dataset = load_dataset(args.dataset, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter indices if requested
    indices = list(range(len(dataset)))
    if args.sample_range:
        indices = []
        try:
            parts = args.sample_range.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start, end))
                else:
                    indices.append(int(part))
        except ValueError:
            print("Error: Invalid sample_range format.")
            return
        # Validate
        indices = [i for i in indices if 0 <= i < len(dataset)]
        if not indices:
            print("No valid indices content.")
            return
    
    subset = dataset.select(indices)
    print(f"Processing {len(subset)} records.")

    # Initialize vLLM
    print(f"Initializing vLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        dtype="bfloat16"
    )
    
    sampling_params = SamplingParams(
        temperature=0.3, # Low temp for deterministic rewriting
        max_tokens=2048,
        stop=["ORIGINAL PROBLEM:", "REWRITTEN PROBLEM:"] # Stop tokens to prevent hallucination loops
    )
    
    # Prepare Prompts
    prompts = []
    metadata = []
    
    for i, example in enumerate(subset):
        original_idx = indices[i]
        problem_text = example.get('problem', example.get('question', ''))
        
        prompt = get_substitution_prompt(problem_text, lower_vars, upper_vars, names_list)
        prompts.append(prompt)
        
        metadata.append({
            "id": example.get('id', original_idx),
            "original_idx": original_idx,
            "original_problem": problem_text,
            "answer": example.get('answer', None) # Keep the answer (it stays same usually)
        })

    # Batch Generate
    print(f"Generating rewrites for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Collect Results
    modified_dataset = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        meta = metadata[i]
        
        entry = {
            "id": meta['id'],
            "original_problem": meta['original_problem'],
            "modified_problem": generated_text,
            "answer": meta['answer'],
            "substitution_meta": {
                 "lower_vars": lower_vars,
                 "upper_vars": upper_vars,
                 "names": names_list
            }
        }
        modified_dataset.append(entry)

    # Save
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.output_file, 'w') as f:
        json.dump(modified_dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved {len(modified_dataset)} modified problems to {args.output_file}")

if __name__ == "__main__":
    main()
