import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, extract_answer, normalize_answer, remove_latex_comments, BASELINE_SYSTEM_PROMPT
# from vllm import LLM, SamplingParams # Moved inside main
from trim_context import trim_context


def main():
    parser = argparse.ArgumentParser(description="Evaluate with Predefined Context Saturation")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Model name/path")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="Dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--context_size", type=int, required=True, help="Target context size in tokens")
    parser.add_argument("--context_type", type=str, choices=['math', 'text'], default='math', help="Distractor type")
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
        context_path = f"experiments/context_{args.context_type}.json"
        
    if not os.path.exists(context_path):
        # Try absolute or relative fix
        if os.path.exists(f"context_{args.context_type}.json"):
             context_path = f"context_{args.context_type}.json"
        else:
             print(f"Error: Context file not found at {context_path}")
             return

    # Initialize vLLM
    llm = None
    sampling_params = None
    tokenizer = None
    
    if not args.dry:
        from vllm import LLM, SamplingParams
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
    trimmed_context = trim_context(context_path, args.model, args.context_size, tokenizer=tokenizer)
    
    # Override System Prompt
    # Ensure standard math system prompt is used, replacing whatever was in context (e.g. history text prompt)
    if trimmed_context and trimmed_context[0]['role'] == 'system':
        if args.context_type == 'text':
            trimmed_context[0]['content'] = BASELINE_SYSTEM_PROMPT
            print(f"Overriding system prompt (was: {trimmed_context[0]['content']})")
    else:
        print("Inserting system prompt...")
        trimmed_context.insert(0, {'role': 'system', 'content': BASELINE_SYSTEM_PROMPT})
    
    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        
    all_inputs = [] # list of token_ids
    metadata = []
    
    # Calculate context tokens once
    context_token_count = len(tokenizer.apply_chat_template(trimmed_context, tokenize=True, add_generation_prompt=False))
    # Get the common context string to save once
    common_context_str = tokenizer.apply_chat_template(trimmed_context, tokenize=False, add_generation_prompt=False)
    
    print(f"Preparing {len(dataset)} examples...")
    
    for i, example in enumerate(dataset):
        cleaned_problem = remove_latex_comments(example['problem'])
        
        # Use util.get_prompts for consistency, using 'baseline' to just get the problem
        user_prompt, system_prompt = get_prompts(cleaned_problem, 'baseline')
        
        # Create full conversation history: context + current user prompt
        # We assume trimmed_context already includes the system prompt if needed
        full_conversation = trimmed_context + [{"role": "user", "content": user_prompt}]
        
        final_input_ids = tokenizer.apply_chat_template(full_conversation, tokenize=True, add_generation_prompt=True)
        
        for sample_idx in range(args.n_samples):
            all_inputs.append(final_input_ids)
            metadata.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "original": user_prompt,
                "ground_truth": example['answer'],
                #"context": tokenizer.decode(context_ids, skip_special_tokens=True) # Context is now part of conversation, hard to decode separately cleanly
            })
    
    # Generate
    print(f"Generating answers for {len(all_inputs)} prompts...")
    
    if not args.dry:
        print("Decoding token sequences to text for inference...")
        decoded_prompts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in all_inputs]
        outputs = llm.generate(decoded_prompts, sampling_params=sampling_params)
        # outputs = llm.generate(prompt_token_ids=all_inputs, sampling_params=sampling_params)
    else:
        print("Dry run: Skipping generation.")
        outputs = []
        decoded_prompts = [] # Initialize for dry run
        # Mock outputs
        class MockOutput:
            def __init__(self, text):
                self.outputs = [type('obj', (object,), {'text': text, 'token_ids': [0]*10})] # Mock 10 tokens
        
        for _ in all_inputs:
            outputs.append(MockOutput("Mock Answer \\boxed{0}"))
            decoded_prompts.append("Dry Run Prompt Mock")
    
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
            "sample_idx": meta.get('sample_idx', 0),
            "output": generated_text,
            "original problem": meta['original'],
            # "full_input": decoded_prompts[i], # REMOVE to save space
            "extracted": extracted,
            "ground_truth": meta['ground_truth'],
            "correct": is_correct,
            "system_prompt": "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n",
            "temperature": 0.7,
            "max_model_length": args.max_model_length,
            "distractor_token_count": context_token_count,
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
    filename = f"results_predef_{args.context_type}_{args.context_size}_{safe_model}_{timestamp}.json"
    
    # New directory structure
    safe_dataset = args.dataset.replace('/', '_')
    dirs = f"context_saturation/predef_cont_results/{safe_model}/{safe_dataset}"
    os.makedirs(dirs, exist_ok=True)
    out_path = os.path.join(dirs, filename)
    
    final_output = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "context_size": args.context_size,
            "context_type": args.context_type,
            "common_context": common_context_str
        },
        "statistics": stats,
        "results": results
    }
    
    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
