import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, extract_answer, normalize_answer, remove_latex_comments
from vllm import LLM, SamplingParams

def load_context_ids(context_file, tokenizer, context_size):
    """
    Loads context from JSON, tokenizes it, and truncates to exactly context_size tokens.
    """
    print(f"Loading context from {context_file}...")
    with open(context_file, 'r') as f:
        context_history = json.load(f)
    
    # We want to tokenize the conversation history as a single block.
    # Depending on the tokenizer/template, we might need apply_chat_template.
    # However, context_history is a list of dicts.
    
    # Attempt to use apply_chat_template on the whole history
    # Note: If the history ends with a user message (which it might in some designs, but usually valid convo is alternating),
    # apply_chat_template handles it.
    
    # We assume standard chat template availability.
    # We set add_generation_prompt=False because this is just history, not the final prompt.
    full_context_text = tokenizer.apply_chat_template(context_history, tokenize=False, add_generation_prompt=False)
    
    # Tokenize
    full_context_ids = tokenizer.encode(full_context_text)
    
    print(f"Full context length: {len(full_context_ids)} tokens.")
    
    if len(full_context_ids) < context_size:
        print(f"Warning: Loaded context ({len(full_context_ids)}) is smaller than requested size ({context_size}). Using full context.")
        return full_context_ids
    
    # Truncate
    truncated_ids = full_context_ids[:context_size]
    print(f"Truncated context to {len(truncated_ids)} tokens.")
    return truncated_ids

def main():
    parser = argparse.ArgumentParser(description="Evaluate with Predefined Context Saturation")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Model name/path")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="Dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--context_size", type=int, required=True, help="Target context size in tokens")
    parser.add_argument("--distractor_type", type=str, choices=['math', 'text'], default='math', help="Distractor type")
    parser.add_argument("--context_file", type=str, default=None, help="Override context file path")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_model_length", type=int, default=65536)
    parser.add_argument("--dry", action="store_true", help="Dry run")
    
    args = parser.parse_args()
    
    # Determine context file
    if args.context_file:
        context_path = args.context_file
    else:
        # Default paths based on type
        # Assuming run from root of project
        context_path = f"experiments/context_{args.distractor_type}.json"
        
    if not os.path.exists(context_path):
        # Try absolute or relative fix
        if os.path.exists(f"context_{args.distractor_type}.json"):
             context_path = f"context_{args.distractor_type}.json"
        else:
             print(f"Error: Context file not found at {context_path}")
             return

    # Initialize vLLM
    llm = None
    sampling_params = None
    tokenizer = None
    
    if not args.dry:
        print(f"Initializing vLLM with model: {args.model}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.num_gpus,
            trust_remote_code=True,
            max_model_len=args.max_model_length,
            dtype="bfloat16"
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_model_length) 
    
        tokenizer = llm.get_tokenizer()
    else:
        print("Dry run: Skipping vLLM initialization. Loading tokenizer from huggingface...")
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except:
            print("Failed to load specific tokenizer, falling back to gpt2 for dry run token counting.")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load and Truncate Context
    context_ids = load_context_ids(context_path, tokenizer, args.context_size)
    
    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        
    all_inputs = [] # list of token_ids
    metadata = []
    
    print(f"Preparing {len(dataset)} examples...")
    
    for i, example in enumerate(dataset):
        cleaned_problem = remove_latex_comments(example['problem'])
        
        # Use util.get_prompts for consistency, using 'baseline' to just get the problem
        user_prompt, system_prompt = get_prompts(cleaned_problem, 'baseline')
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        problem_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        
        # Remove BOS from problem_ids if it matches context_ids[0] (likely BOS)
        # Assuming BOS is usually the first token.
        if tokenizer.bos_token_id is not None:
             if len(problem_ids) > 0 and problem_ids[0] == tokenizer.bos_token_id:
                  problem_ids = problem_ids[1:]
        
        final_input_ids = context_ids + problem_ids
        
        all_inputs.append(final_input_ids)
        metadata.append({
            "id": example.get('id', i),
            "original": user_prompt,
            "ground_truth": example['answer']
        })
    
    # Generate
    print(f"Generating answers for {len(all_inputs)} prompts...")
    
    if not args.dry:
        outputs = llm.generate(prompt_token_ids=all_inputs, sampling_params=sampling_params)
    else:
        print("Dry run: Skipping generation.")
        outputs = []
        # Mock outputs
        class MockOutput:
            def __init__(self, text):
                self.outputs = [type('obj', (object,), {'text': text, 'token_ids': [0]*10})] # Mock 10 tokens
        
        for _ in all_inputs:
            outputs.append(MockOutput("Mock Answer \\boxed{0}"))
    
    results = []
    stats = {"correct": 0, "total": 0, "failures": 0}
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        output_token_ids = output.outputs[0].token_ids
        output_len = len(output_token_ids)
        
        meta = metadata[i]
        
        extracted = extract_answer(generated_text)
        is_correct = False
        if extracted:
             is_correct = normalize_answer(extracted) == normalize_answer(meta['ground_truth'])
        
        
        results.append({
            "id": meta['id'],
            "output": generated_text,
            "extracted": extracted,
            "ground_truth": meta['ground_truth'],
            "correct": is_correct,
            # Enhanced metadata
            "system_prompt": "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n", # Hardcoded from util.BASELINE_SYSTEM_PROMPT
            "temperature": 0.7,
            "max_model_length": args.max_model_length,
            "distractor_token_count": len(context_ids),
            "model_output_token_count": output_len
        })
        
        stats["total"] += 1
        if is_correct: stats["correct"] += 1
        else: stats["failures"] += 1
        
    print("\n=== Evaluation Results ===")
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace('/', '_')
    filename = f"results_predef_{args.distractor_type}_{args.context_size}_{safe_model}_{timestamp}.json"
    
    # New directory structure
    safe_dataset = args.dataset.replace('/', '_')
    dirs = f"experiments/context_saturation/predef_cont_results/{safe_model}/{safe_dataset}"
    os.makedirs(dirs, exist_ok=True)
    out_path = os.path.join(dirs, filename)
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
