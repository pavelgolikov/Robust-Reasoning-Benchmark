import argparse
import os
import sys
import os

# Fix path to include 'analysis' so 'variables' can be imported by util
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'analysis'))

import json
import time
import random
import re
import io
import contextlib
import traceback
from datasets import load_dataset
from util import get_prompts, extract_answer, normalize_answer

# ======================================================
# PART 1: THE PARSER (Identification)
# ======================================================
def extract_python_code(response_text):
    """
    Uses Regex to find content between ```python and ``` tags.
    Returns None if no code block is found.
    """
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        return match.group(1) # Return just the code inside
    return None

# ======================================================
# PART 2: THE EXECUTOR (Running on CPU)
# ======================================================
def execute_python_code(code_str, state_dict):
    """
    Executes code. Captures stdout. 
    If it crashes, returns Partial Output + Traceback.
    """
    output_capture = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(output_capture):
            # Safe-guarding: In a real sandboxed env, we'd be more careful.
            # Here we just execute in the provided state_dict.
            exec(code_str, state_dict)
            
        # If we get here, no error occurred.
        result = output_capture.getvalue()
        if not result:
            return "[Code ran successfully, but produced no output.]"
        return result

    except Exception:
        # 1. Get whatever was printed BEFORE the crash
        partial_output = output_capture.getvalue()
        
        # 2. Get the full traceback
        error_trace = traceback.format_exc()
        
        # 3. Combine them
        return f"{partial_output}\n\n--- EXECUTION ERROR ---\n{error_trace}"

# ======================================================
# PART 3: THE AGENT LOOP
# ======================================================
def run_agent_turn(system_prompt, user_input, llm, tokenizer, sampling_params, dry_run=False):
    # 1. Initialize History
    history = [ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_input} ]
    # 2. Initialize Memory
    agent_memory = {}
    final_response = ""
    # Run for up to 5 steps
    for step in range(0, 1):
        # A. CALL THE MODEL
        if not dry_run:
            formatted_prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            outputs = llm.generate([formatted_prompt], sampling_params)
            model_msg = outputs[0].outputs[0].text
        else:
            model_msg = "Final Answer: 42"
        # print(f"\n[AI Step {step}]: {model_msg[:100]}...")
        # Add AI response to history
        history.append({"role": "assistant", "content": model_msg})

        # # Check for Final Answer
        # if "Final Answer" in model_msg:
        #      final_response = model_msg
        #      # print("--- AGENT FINISHED ---")
        #      break

        # # B. PARSE: Is there code?
        # code_block = extract_python_code(model_msg)
        
        if code_block:
            # print(f"\n[Extracted Code]: {code_block[:50]}...")
            # C. EXECUTE
            execution_result = execute_python_code(code_block, agent_memory)
            # print(f"[Execution Output]: {execution_result[:50]}...")
            # D. FEEDBACK
            tool_output_msg = f"Observation:\n{execution_result}"
            history.append({"role": "user", "content": tool_output_msg})
        else:
            # If no code and no Final Answer, we might stop or continue. 
            # The loop continues to give it a chance to do something else, 
            # or we could force a stop if we assume it must output code or answer.
            # For now, let's treat it as a potential continuation/reasoning step unless we hit max steps.
            pass
            
    # If loop finishes without break, take the last message
    if not final_response:
        final_response = history[-1]['content']
        
    return final_response


def main():
    parser = argparse.ArgumentParser(description="Evaluate manual agent implementation")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Path/Name of the model")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of experiment names")
    parser.add_argument("--num_gpus", type=int, default=2, help="Num GPUs.")
    parser.add_argument("--max_model_length", type=int, default=32000, help="Max model length for vLLM")
    parser.add_argument("--dry", action="store_true", help="Dry run - do not load model")
    parser.add_argument("--num_distractors", type=int, default=32, help="Number of distractors for split_indices")
    parser.add_argument("--chat", action="store_true", help="Run in chat mode")

    args = parser.parse_args()

    # Determine experiment names
    if args.names == 'all':
        experiment_names = [ 'context_saturation', 'interleaved_context_line', 'interleaved_context_word',
        'not_not_yot', 'opposites', 'sentence_reversal', 'word_reversal', 'word_split_swap', 'wrappers', 'split_reversal' ]
    else:
        experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]

    print(f"Running experiments: {experiment_names}")

    # Initialize vLLM
    llm = None
    sampling_params = None
    tokenizer = None
    
    if not args.dry:
        print(f"Initializing vLLM with model: {args.model}")
        try:
            from vllm import LLM, SamplingParams
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.num_gpus,
                trust_remote_code=True,
                max_model_len=args.max_model_length,
                dtype="bfloat16"
            )
            # Stop tokens are crucial for ReAct to stop generating after producing code block or before user role
            # The user snippet used stop=["Observation:"].
            # We should probably also stop at "Observation:" and maybe user role tokens.
            # Qwen uses specific template, but "Observation:" is the standard ReAct stop.
            sampling_params = SamplingParams(
                temperature=0.0, 
                max_tokens=2048,
                stop=["Observation:"]
            )
            tokenizer = llm.get_tokenizer()
        except Exception as e:
            print(f"Failed to initialize vLLM: {e}")
            exit(1)
    else:
        print("Dry run: Skipping vLLM initialization.")

    random.seed(args.seed)

    # # Chat Mode Logic
    # if args.chat:
    #     if args.dry:
    #         print("Chat mode requires a loaded model. Dry run is not useful here (agent is None).")
    #         # We can mock it for testing flow though
    #         class MockAgent:
    #             def run(self, x): return f"Mock Response to: {x}"
    #         agent = MockAgent()
    #     else:
    #          # Initialize a single Global Agent for Chat
    #         try:
    #             agent = CodeAgent(tools=[], model=model_engine, add_base_tools=True)
    #         except Exception as e:
    #             print(f"Failed to initialize Agent for chat: {e}")
    #             exit(1)
        
    #     chat_loop(agent)
    #     return # Exit after chat

    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Load Variables (Copied logic from evaluate.py)
    extracted_vars = {}
    if 'split_indices' in experiment_names:
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

    # System Prompt Template
#     BASE_SYSTEM_PROMPT = """You are a Python-Equipped Math Agent.
# 1. If the problem is obfuscated, write Python to decode it first.
# 2. Once decoded, write Python to solve the math.
# 3. Output your code in markdown: ```python ... ```
# 4. You MUST print() your results to see them.
# 5. When done, output "Final Answer: [value]".
# """
    BASE_SYSTEM_PROMPT = get_system_prompt("base")

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
            
            user_prompt_content, _ = get_prompts(
                example['problem'], 
                exp_name, 
                extra_context, 
                variables=current_vars,
                seed=args.seed, 
                num_distractors=args.num_distractors,
                decode_find_only=False
            )
            ground_truth = example['answer']

            print(f"  Sample {i} (ID: {prob_id})...", end="", flush=True)
            
            try:
                # Combine our base system prompt with specific instructions if needed, 
                # or just use our base system prompt and append the specific prompt as user context?
                # The get_prompts returns a system prompt too. 
                # ReAct agent needs specific instructions on HOW to behave (the 5 rules).
                # So we should combine them.
                
                # We will ignore the system prompt from get_prompts for the agent's behavior instructions,
                # BUT we might need the specific decoding instructions if they were in the system prompt.
                # However, looking at util.py, the system prompts are usually: "You are a helpful math assistant... User query contains..."
                # We should append that context to the user prompt or merge it.
                # Let's append the technique description to the user prompt to ensure the agent knows what to look for,
                # but keep the Base System Prompt as the main system instruction.
                
                _, technique_system_prompt = get_prompts(
                     example['problem'], exp_name, extra_context, current_vars, args.seed, args.num_distractors, False
                )
                
                # Extract the description part from technique_system_prompt (remove "You are a helpful math assistant.")
                technique_description = technique_system_prompt.replace("You are a helpful math assistant.", "").strip()
                
                full_user_input = f"{technique_description}\n\nTask:\n{user_prompt_content}"

                final_output = run_agent_turn(BASE_SYSTEM_PROMPT, full_user_input, llm, tokenizer, sampling_params, args.dry)
                
                print(" Done.")

            except Exception as e:
                print(f" Error: {e}")
                final_output = f"ERROR: {e}"

            # Grade
            try:
                extracted = extract_answer(final_output)
                is_correct = normalize_answer(extracted) == normalize_answer(ground_truth)
            except Exception:
                extracted = "ERROR_PARSING"
                is_correct = False
            
            entry = {
                "id": example.get('id', i),
                "system_prompt": BASE_SYSTEM_PROMPT,
                "user_prompt": full_user_input,
                "original_transformed": user_prompt_content,
                "unmodified_original": example['problem'],
                "ground_truth": ground_truth,
                "output": final_output,
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
        
        run_id = f"{safe_model_name}_{safe_dataset_name}_{exp_name}_s{args.seed}_{timestamp}_MANUAL_AGENT"
        json_file = os.path.join(final_output_dir, f"{run_id}.json")
        
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {json_file}")


def get_system_prompt(exp_name):
    return "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}.\n"


if __name__ == "__main__":
    main()
