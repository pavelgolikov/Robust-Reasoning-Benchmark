import os
import json
import re
import time
from opposites.transformation import apply_opposite_semantic_remapping
from opposites_not.transformation import apply_opposites_not_yot
from interleaved_context.transformation import apply_interleaved_context
from interleaved_substitutions.transformation import apply_interleaved_substitutions
from wrappers.transformation import apply_wrappers
from variables.transformation import apply_variables
from context_saturation.transformation import apply_context_saturation
from not_not_yot.transformation import apply_not_not_yot
from word_split_swap.transformation import apply_word_split_swap
from word_reversal.transformation import apply_word_reversal
from sentence_reversal.transformation import apply_sentence_reversal

import multiprocessing
import os
# Force 'spawn' to avoid CUDA re-initialization errors
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import random
import argparse
from datasets import load_dataset
from opposites.transformation import apply_opposites
import nltk

# Ensure NLTK data (WordNet) is available
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

def get_prompts(problem, name, extra_context=None, variables=None, seed=None, num_distractors=None):
    # modify problem according to experiment name
    if name == 'baseline':
        user_prompt = problem
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n"
        return user_prompt, system_prompt
    elif name == 'not_not_yot':
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
Yot means the opposite of not.\n"
        user_prompt = apply_not_not_yot(problem)
        return user_prompt, system_prompt
    elif name == 'word_split_swap':
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
All words (words are defined as sequences of symbols separated by spaces) in user query have been modified as follows.\n\
Every word is first split into 2 parts. If the word has even number of symbols, it is split into 2 equal parts in the middle. \n\
If the word has odd number of symbols, the first part has one symbol less than the second part. \n\
After splitting, the 2 parts were swapped. Punctuation marks adjacent to words are counted as word symbols.\n"
        user_prompt = apply_word_split_swap(problem)
        return user_prompt, system_prompt
    elif name == 'word_reversal':
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
The order of words (words are defined as sequences of symbols separated by spaces) in each sentence of user query has been reversed.\n"
        user_prompt = apply_word_reversal(problem)
        return user_prompt, system_prompt
    elif name == 'sentence_reversal':
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
The order of sentences in the user query has been reversed. Sentences are defined as sequences of symbols separated by periods.\n"
        user_prompt = apply_sentence_reversal(problem)
        return user_prompt, system_prompt
    elif name == 'opposites':
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.\n"
        user_prompt = apply_opposites(problem, k=1)
        return user_prompt, system_prompt
    elif name == 'opposites_not':
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of user query.\n"
        user_prompt = apply_opposites_not_yot(problem, k_opp=1)
        return user_prompt, system_prompt
    elif name == 'interleaved_context':
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
User query will consist of two problems - A and B, whose statements are interleaved.\n\
You need to solve only problem A. If one problem statement is shorter than the other,\n\
the empty lines resulting from the shorter problem statement will be filled with the\n\
shorter problem statement repeated from the beginning."
        if extra_context is None:
            user_prompt = "Error: Missing extra context for interleaved transformation"
        else:
            user_prompt = apply_interleaved_context(problem, extra_context)
        return user_prompt, system_prompt
    elif name == 'interleaved_substitutions':
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
User query will consist of two problems - A and B, whose statements are interleaved.\n\
You need to solve only problem A. If one problem statement is shorter than the other,\n\
the empty lines resulting from the shorter problem statement will be filled with the\n\
shorter problem statement repeated from the beginning. On top of that, some words in problem A\n\
are remapped. The remappings are defined inside 'defyn{}' block in the middle of user query.\n"
        if extra_context is None:
            user_prompt = "Error: Missing extra context for interleaved transformation"
            exit(1)
        else:
            user_prompt = apply_interleaved_substitutions(problem, extra_context, k=1)
        return user_prompt, system_prompt
    elif name == 'wrappers':
        system_prompt = "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
There will be terms remapped in the user query. The remappings are defined inside 'defyn{}' block in the middle of the user query.\n"
        user_prompt = apply_wrappers(problem, k=1)
        return user_prompt, system_prompt
    elif name == 'variables':
        system_prompt = "You are a helpful math assistant.\n\
Your goal is to identify important 'load-bearing' terms in a math problem that we will later target for redefinition.\n"
        user_prompt = apply_variables(problem)
        return user_prompt, system_prompt
    elif name == 'context_saturation':
        system_prompt = """You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n\
You will be given multiple problem statements. Problem index in the form [[ProblemK]] will be embedded in each problem\n\
statement itself, but will be split into 2 parts. Each of the two parts will form its own sentence and will be\n\
placed between sentences of the problem statement.\n\
For example, two valid parts are: "[[Pro"     "blemK]]" and "[[P"    "roblemK]]". The two parts can be placed in reverse\n\
order in the problem statement, for example, "blemK]]" first and "[[Pro" second. The index of the problem you are to solve\n\
will be indicated in the middle of user query. Problems do not depend on each other.\n
""" 
        user_prompt = apply_context_saturation(problem, num_distractors, seed=seed, problem_variables=variables)
        return user_prompt, system_prompt
    else:
        return 'Not Implemented', ''

def extract_answer(text):
    if not text:
        return None
    
    # Priority 1: Boxed
    boxed_pattern = r"\\boxed\s*\{([^}]+)\}"
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()
        
    # Priority 2: Explicit answer statement
    answer_pattern = r"(?:The answer is|result is|so|equals)\s*[:=]?\s*(\d{1,4})(?:\.|,|\s|$)"
    matches = re.findall(answer_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1]
    
    return None

def normalize_answer(ans):
    if ans is None:
        return ""
    digits = "".join(filter(str.isdigit, str(ans)))
    if not digits:
        return ""
    return str(int(digits))

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on AIME dataset (Killarney/vLLM)")
    parser.add_argument("--model", type=str, default="NONE", help="Path/Name of the model to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--name", type=str, default='baseline', help="Name of the experiment")
    parser.add_argument("--dry", action="store_true", help="Dry run - do not evaluate, only produce prompts")
    parser.add_argument("--num_distractors", type=int, default=30, help="Number of distractors for split_indices")
    args = parser.parse_args()

    # Load extracted variables if needed
    extracted_vars = {}
    if args.name == 'split_indices':
        try:
            vars_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variables', 'extracted_terms_by_problem.json')
            with open(vars_path, 'r') as f:
                extracted_vars = json.load(f)
            # replace all spaces with underscores in each variable name
            for k, v in extracted_vars.items():
                extracted_vars[k] = [x.replace(" ", "_") for x in v]
            print("Variables loaded:", extracted_vars)
        except Exception as e:
            print(f"Warning: Failed to load extracted variables: {e}")

    if not args.dry:
        print(f"Initializing vLLM with model: {args.model}")
        from vllm import LLM, SamplingParams
        max_model_length = 32000
        llm = LLM(
            model=args.model,
            tensor_parallel_size=2,
            trust_remote_code=True,
            max_model_len=max_model_length,
            dtype="bfloat16"
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=max_model_length)

    random.seed(args.seed)
    
    # Load Dataset
    print("Loading AIME 2024 dataset...")
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        
    print(f"Starting Evaluation on {len(dataset)} examples. Seed={args.seed}. Samples per problem={args.n_samples}")

    prompts = []
    metadata = []

    if not args.dry:
        tokenizer = llm.get_tokenizer()

    for i, example in enumerate(dataset):
        extra_context = None
        
        # Pre-process problem to remove empty lines
        problem_text = example['problem']
        problem_text = "\n".join([line for line in problem_text.splitlines() if line.strip()])
        
        if args.name in ['interleaved_context', 'interleaved_substitutions']:
            # Use next problem as context, wrapping around to the first for the last problem
            next_idx = (i + 1) % len(dataset)
            extra_context = dataset[next_idx]['problem']
            
        prob_id = str(example.get('id', i))
        current_vars = extracted_vars.get(prob_id) if extracted_vars else None
        
        user_prompt, system_prompt = get_prompts(problem_text, args.name, extra_context, variables=current_vars,
                                                seed=args.seed, num_distractors=args.num_distractors)
        ground_truth = example['answer']
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if not args.dry:
            # Format prompt
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = messages
        
        # Create n independent samples for this problem
        for sample_idx in range(args.n_samples):
            prompts.append(formatted_prompt)
            metadata.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "original": user_prompt,
                "ground_truth": ground_truth
            })

    # Generate
    print(f"Generating responses for {len(prompts)} prompts...")
    if not args.dry:
        outputs = llm.generate(prompts, sampling_params)
    else:
        outputs = [''] * len(prompts)

    # Process Results
    results = []
    correct_count = 0
    total = 0
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = args.model.replace('/', '_').replace(' ', '_')
    run_id = f"{safe_name}_{args.name}_s{args.seed}_{timestamp}"
    
    # Save to [experiment_name]/results logic
    base_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.join(base_dir, args.name)
    final_output_dir = os.path.join(experiment_dir, "results")
    
    os.makedirs(final_output_dir, exist_ok=True)
    json_file = os.path.join(final_output_dir, f"{run_id}.json")

    for i, output in enumerate(outputs):
        if not args.dry:
            generated_text = output.outputs[0].text
        else:
            generated_text = 'placeholder output from dry run'
        meta = metadata[i]
        
        extracted = extract_answer(generated_text)
        is_correct = normalize_answer(extracted) == normalize_answer(meta['ground_truth'])
        
        if is_correct:
            correct_count += 1
            
        result_entry = {
            "id": meta['id'],
            "system_prompt": system_prompt,
            "original": meta['original'],
            "ground_truth": meta['ground_truth'],
            "output": generated_text,
            "extracted": extracted,
            "correct": is_correct
        }
        results.append(result_entry)
        total += 1
        
    accuracy = correct_count / total if total > 0 else 0
    print(f"\nEvaluation Complete. Accuracy: {accuracy:.2%} ({correct_count}/{total})")
    print(f"Results saved to: {json_file}")
    
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
