
import argparse
import os
import json
import time
import random
try:
    from smolagents import CodeAgent, VLLMModel
except ImportError:
    print("smolagents not found. Install with `pip install smolagents`")
    exit(1)

from datasets import load_dataset
from util import get_prompts, extract_answer, normalize_answer

def pre_saturate_context(agent, question):
    print(f"pre_saturation not implemented. Question: {question[:50]}...")

def chat_loop(agent):
    print("--- Starting Interactive Chat with Agent (type 'exit' to quit) ---")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            
            # Run agent
            # Note: agent.run() accumulates history automatically.
            response = agent.run(user_input)
            print(f"Agent: {response}")
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"Error during chat: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple experiments on AIME dataset using SmolAgents")
    parser.add_argument("--model", type=str, default="GAIR/LIMO-v2", help="Path/Name of the model to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of experiment names")
    parser.add_argument("--num_distractors", type=int, default=32, help="Number of distractors for split_indices")
    parser.add_argument("--num_gpus", type=int, default=2, help="Num GPUs.")
    parser.add_argument("--max_model_length", type=int, default=32000, help="Max model length for vLLM")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--dry", action="store_true", help="Dry run - do not load model")
    parser.add_argument("--pre_saturate", action="store_true", help="If set, run pre-saturation routine on agent before prompt.")
    parser.add_argument("--chat", action="store_true", help="Enter interactive chat mode with the agent.")
    
    args = parser.parse_args()

    # Determine experiment names
    if args.names == 'all':
        experiment_names = [ 'context_saturation', 'interleaved_context_line', 'interleaved_context_word',
        'not_not_yot', 'opposites', 'sentence_reversal', 'word_reversal', 'word_split_swap', 'wrappers', 'split_reversal' ]
    else:
        experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]

    print(f"Running experiments: {experiment_names}")

    if not args.dry:
        # Initialize VLLM Model via smolagents
        print(f"Initializing VLLMModel with model: {args.model}")
        try:
            model_engine = VLLMModel(
                model_id=args.model,
                model_kwargs={
                    "tensor_parallel_size": args.num_gpus,
                    "trust_remote_code": True,
                    "max_model_len": args.max_model_length,
                    "dtype": "bfloat16"
                }
            )
        except Exception as e:
            print(f"Failed to initialize VLLMModel: {e}")
            exit(1)
    else:
        print("Dry run: Skipping Model/Agent initialization.")
        model_engine = None

    random.seed(args.seed)

    # Chat Mode Logic
    if args.chat:
        if args.dry:
            print("Chat mode requires a loaded model. Dry run is not useful here (agent is None).")
            # We can mock it for testing flow though
            class MockAgent:
                def run(self, x): return f"Mock Response to: {x}"
            agent = MockAgent()
        else:
             # Initialize a single Global Agent for Chat
            try:
                agent = CodeAgent(tools=[], model=model_engine, add_base_tools=True)
            except Exception as e:
                print(f"Failed to initialize Agent for chat: {e}")
                exit(1)
        
        chat_loop(agent)
        return # Exit after chat

    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"Starting Agent Evaluation on {len(dataset)} examples. Seed={args.seed}. Mode={'Pre-Saturate' if args.pre_saturate else 'Standard'}")

    # Load extracted variables if needed (if split_indices is in list)
    # Similar logic to evaluate.py
    extracted_vars = {}
    if 'split_indices' in experiment_names: # referencing split_indices logic from evaluate.py even if not in standard list
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            vars_path = os.path.join(base_dir, 'variables', 'extracted_terms_by_problem.json')
            if os.path.exists(vars_path):
                with open(vars_path, 'r') as f:
                    extracted_vars = json.load(f)
                for k, v in extracted_vars.items():
                    extracted_vars[k] = [x.replace(" ", "_") for x in v]
                print("Variables loaded for split_indices.")
        except Exception:
            pass

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model.replace('/', '_').replace(' ', '_')
    safe_dataset_name = args.dataset.replace('/', '_')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for exp_name in experiment_names:
        print(f"\nEvaluating Experiment: {exp_name}")
        
        results = []
        stats = {"correct": 0, "total": 0, "failures": 0}

        for i, example in enumerate(dataset):
            # Prepare Prompt
            extra_context = None
            if exp_name in ['interleaved_context_word', 'interleaved_context_line', 'interleaved_substitutions']:
                next_idx = (i + 1) % len(dataset)
                extra_context = dataset[next_idx]['problem']
            
            prob_id = str(example.get('id', i))
            current_vars = extracted_vars.get(prob_id) if extracted_vars else None
            
            user_prompt, system_prompt_text = get_prompts(
                example['problem'], 
                exp_name, 
                extra_context, 
                variables=current_vars,
                seed=args.seed, 
                num_distractors=args.num_distractors,
                decode_find_only=False 
            )
            ground_truth = example['answer']

            # Run Agent
            # CodeAgent takes the task (prompt).
            # It usually returns the final answer directly as the result of run().
            full_prompt = f"{system_prompt_text}\n\nTask:\n{user_prompt}"
            
            print(f"  Sample {i} (ID: {prob_id})...", end="", flush=True)
            try:
                if not args.dry:
                    # Initialize Fresh Agent per sample
                    agent = CodeAgent(
                        tools=[],
                        model=model_engine,
                        system_prompt=system_prompt_text,
                        add_base_tools=True
                    )
                    
                    if args.pre_saturate:
                        pre_saturate_context(agent, example['problem'])

                    # Run the agent
                    output = agent.run(full_prompt)
                    generated_text = str(output)
                else:
                    if args.pre_saturate:
                        pre_saturate_context(None, example['problem'])
                    generated_text = "DRY_RUN_OUTPUT"
                print(" Done.")

            except Exception as e:
                print(f" Error: {e}")
                generated_text = f"ERROR: {e}"

            # Grade
            try:
                extracted = extract_answer(generated_text)
                is_correct = normalize_answer(extracted) == normalize_answer(ground_truth)
            except Exception:
                extracted = "ERROR_PARSING"
                is_correct = False
                
            entry = {
                "id": example.get('id', i),
                "system_prompt": system_prompt_text,
                "original": user_prompt,
                "unmodified_original": example['problem'],
                "ground_truth": ground_truth,
                "output": generated_text,
                "extracted": extracted,
                "correct": is_correct
            }
            results.append(entry)
            
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1
            else:
                stats["failures"] += 1
        
        # Save Results per experiment
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"Results for {exp_name}: Accuracy {acc:.2%} ({stats['correct']}/{stats['total']})")
        
        experiment_dir = os.path.join(base_dir, exp_name)
        final_output_dir = os.path.join(experiment_dir, "results", safe_model_name, safe_dataset_name)
        os.makedirs(final_output_dir, exist_ok=True)
        
        run_id = f"{safe_model_name}_{safe_dataset_name}_{exp_name}_s{args.seed}_{timestamp}_AGENT"
        json_file = os.path.join(final_output_dir, f"{run_id}.json")
        
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {json_file}")






if __name__ == "__main__":
    main()
