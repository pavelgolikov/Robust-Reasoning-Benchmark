import argparse
import os
import sys
import base64
import json
import time
import random
import re
import io
import contextlib
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# Fix path to include 'analysis' so 'variables' can be imported by util
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'analysis'))

from datasets import load_dataset
from util import get_prompts, extract_answer, normalize_answer

# ======================================================
# PART 1: HELPER FUNCTIONS (Parser & Executor)
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
# PART 2: AGENT STATE MANAGEMENT
# ======================================================

@dataclass
class AgentState:
    id: str  # Unique ID (e.g., "60_sample_0")
    problem_id: str
    sample_idx: int
    experiment_name: str
    
    # Prompts & Data
    original_problem: str
    ground_truth: str
    system_prompt_static: str # The Protocol Prompt
    user_prompt_content: str  # The Transformed Input (for logging)
    
    # Dynamic State
    history: List[Dict[str, str]]
    memory: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_done: bool = False
    final_output: str = ""
    extracted_answer: str = ""
    is_correct: bool = False
    step_count: int = 0
    max_steps: int = 10

    def get_vllm_prompt(self, tokenizer):
        return tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)

# ======================================================
# PART 3: BATCHED EXECUTION LOOP
# ======================================================

def run_batch_execution(agents: List[AgentState], llm, tokenizer, sampling_params):
    """
    Runs the agent loop for multiple agents in parallel (batched inference).
    """
    active_agents = [a for a in agents if not a.is_done]
    
    while active_agents and any(not a.is_done for a in active_agents):
        # 1. Prepare Prompts
        current_batch_agents = [a for a in active_agents if not a.is_done]
        if not current_batch_agents:
            break
            
        prompts = []
        # 1. Prepare Prompts & Truncate
        max_input_tokens = 65536 - 4096 - 200 # Leave room for generation + small buffer
        
        for agent in current_batch_agents:
            # Recursive truncation loop
            while True:
                prompt_str = agent.get_vllm_prompt(tokenizer)
                token_ids = tokenizer.encode(prompt_str)
                
                if len(token_ids) <= max_input_tokens:
                    prompts.append(prompt_str)
                    break
                
                # Truncation needed
                # Strategy: Keep System (0) and Original Task (1). Remove oldest history (2).
                if len(agent.history) > 2:
                    # Remove the oldest message after the initial setup
                    # We remove one by one. Ideally pairs, but one by one is safe enough for length reduction
                    removed = agent.history.pop(2)
                    # print(f"  [Truncation] Agent {agent.id} removed old message ({len(removed['content'])} chars) to fit context.")
                else:
                    # If we are down to just System + User and it still doesn't fit, we are in trouble.
                    # We can't prune anymore without losing the task.
                    # This implies the user prompt itself is massive.
                    # Fallback: Just let it fail in the try/catch block or hard truncate the prompt?
                    # For now, let's break and let the existing error handler catch the length error.
                    prompts.append(prompt_str)
                    break
        
        print(f"\n[Batch Step] Generating for {len(current_batch_agents)} agents...")
        
        # 2. Batch Generation with Retry/Recovery Logic
        try:
            # Atomic call. If one prompt is invalid, this raises ValueError/IndexError.
            outputs = llm.generate(prompts, sampling_params)
            
            # If successful, map outputs
            for agent, out_obj in zip(current_batch_agents, outputs):
                if not out_obj.outputs:
                    agent.final_output = "ERROR: vLLM returned no output"
                    agent.is_done = True
                    print(f"  [Agent {agent.id}] Failed: No output.")
                    continue
                    
                response_text = out_obj.outputs[0].text
                agent.step_count += 1
                agent.history.append({"role": "assistant", "content": response_text})
                
                # Check for Code
                code_block = extract_python_code(response_text)
                if code_block:
                    execution_result = execute_python_code(code_block, agent.memory)
                    tool_msg = f"Observation:\n{execution_result}"
                    agent.history.append({"role": "user", "content": tool_msg})
                else:
                    if "\\boxed{" in response_text:
                        agent.final_output = response_text
                        agent.is_done = True
                    elif agent.step_count >= agent.max_steps:
                        agent.final_output = response_text
                        agent.is_done = True
                    else:
                        pass # Implicit continue

        except BaseException as e:
            # Catch Validation/Context Errors (ValueError) or others
            print(f"\n[BATCH ERROR] Generation failed: {e}")
            
            # 1. Identify which agent(s) caused the error (e.g. prompt too long)
            # We must token-check to find the culprit(s).
            try:
                # max_len = llm.llm_engine.model_config.max_model_len
                max_len = 65536
            except:
                max_len = 65536

            agents_marked_failed = 0
            for agent, prompt in zip(current_batch_agents, prompts):
                try:
                    toks = tokenizer.encode(prompt)
                    if len(toks) > max_len:
                        print(f"  [Error Handler] KILLED Agent {agent.id}: Prompt length {len(toks)} > {max_len}")
                        agent.final_output = f"ERROR: Prompt length {len(toks)} > {max_len}"
                        agent.is_done = True
                        agents_marked_failed += 1
                except Exception:
                    # If tokenization itself fails, kill it too
                    agent.final_output = "ERROR: Tokenization check failed"
                    agent.is_done = True
                    agents_marked_failed += 1
            
            # 2. If NO specific agents were found to be 'bad' by length check, 
            # implies the error was something else affecting the whole batch or engine.
            # In this case, we MUST stop everyone to prevent infinite loop of crashing.
            if agents_marked_failed == 0:
                print("  [Error Handler] Could not identify specific bad agents (lengths ok). Stopping THIS batch to prevent infinite loop.")
                for agent in current_batch_agents:
                    if not agent.is_done:
                        agent.final_output = f"CRITICAL BATCH ERROR: {e}"
                        agent.is_done = True
            else:
                print(f"  [Error Handler] Killed {agents_marked_failed} agents. Survivors will continue in next loop iteration.")

        # Filter active for next loop
        active_agents = [a for a in agents if not a.is_done]

# ======================================================
# PART 4: MAIN SETUP
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate manual agent implementation (Batched)")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Path/Name of the model")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of experiment names")
    parser.add_argument("--num_gpus", type=int, default=2, help="Num GPUs.")
    parser.add_argument("--max_model_length", type=int, default=65536, help="Max model length for vLLM")
    parser.add_argument("--num_distractors", type=int, default=32, help="Number of distractors for split_indices")

    args = parser.parse_args()

    # Prep Experiments
    if args.names == 'all':
        experiment_names = [ 'context_saturation', 'interleaved_context_line', 'interleaved_context_word',
        'not_not', 'opposites', 'sentence_reversal', 'word_reversal', 'word_split_swap', 'wrappers', 'split_reversal' ]
    else:
        experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]

    print(f"Running experiments: {experiment_names}")

    # Initialize vLLM
    llm = None
    sampling_params = None
    tokenizer = None
    
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
        # Use same params as before
        sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=4096,
            stop=["Observation:"]
        )
        tokenizer = llm.get_tokenizer()
    except Exception as e:
        print(f"Failed to initialize vLLM: {e}")
        exit(1)

    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Load Variables
    extracted_vars = {}
    if 'split_indices' in experiment_names:
        try:
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


    # ------------------------------------------------------
    # MAIN LOOP: Iterate Experiments -> Create Agents -> Batch Run
    # ------------------------------------------------------
    for exp_name in experiment_names:
        print(f"\nEvaluating Experiment: {exp_name}")
        
        # 1. Initialize ALL Agents for this experiment
        agents = []
        random.seed(args.seed) # Reset seed for consistency per experiment
        
        for i, example in enumerate(dataset):
            prob_id = str(example.get('id', i))
            
            # Prepare Prompt components
            extra_context = None
            if exp_name in ['interleaved_context_word', 'interleaved_context_line', 'interleaved_substitutions']:
                next_idx = (i + 1) % len(dataset)
                extra_context = dataset[next_idx]['problem']
            
            current_vars = extracted_vars.get(prob_id) if extracted_vars else None
            
            # Create K samples
            for sample_idx in range(args.n_samples):
                # Generate specific prompt for this sample (distractors/vars might have randomness if utilized)
                # Note: get_prompts uses random.sample if num_distractors > ... so we should ideally control seed.
                # We can set global seed or pass seed. get_prompts has a seed arg.
                
                try:
                    final_user_prompt, system_prompt = get_prompts(
                        example['problem'], 
                        exp_name, 
                        extra_context, 
                        variables=current_vars,
                        seed=args.seed, # Ensure distinct seed per sample/problem
                        num_distractors=args.num_distractors,
                        agentic=True
                    )
                    
                    # Create Agent
                    agent = AgentState(
                        id=f"{prob_id}_s{sample_idx}",
                        problem_id=prob_id,
                        sample_idx=sample_idx,
                        experiment_name=exp_name,
                        original_problem=example['problem'],
                        ground_truth=example['answer'],
                        system_prompt_static=system_prompt,
                        user_prompt_content=final_user_prompt,
                        history=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": final_user_prompt}
                        ]
                    )
                    agents.append(agent)
                    
                except Exception as e:
                    print(f"Skipping Problem {prob_id} Sample {sample_idx} due to error: {e}")
                    traceback.print_exc()

        print(f"Initialized {len(agents)} agents. Starting Batch Execution...")
        
        # 2. Run Batch
        run_batch_execution(agents, llm, tokenizer, sampling_params)
        
        # 3. Collect Results
        results = []
        stats = {"correct": 0, "total": 0, "failures": 0}
        
        for agent in agents:
            # Grade
            try:
                extracted = extract_answer(agent.final_output)
                is_correct = normalize_answer(extracted) == normalize_answer(agent.ground_truth)
            except Exception:
                extracted = "ERROR_PARSING"
                is_correct = False
            
            agent.is_correct = is_correct
            agent.extracted_answer = extracted
            
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1
            else:
                stats["failures"] += 1
                
            results.append({
                "id": agent.problem_id,
                "sample_idx": agent.sample_idx,
                "output": agent.final_output,
                "extracted": extracted,
                "correct": is_correct,
                "original_problem": agent.original_problem,
                "ground_truth": agent.ground_truth,
                "history_dump": [h['content'] for h in agent.history] # Optional: save full history
            })

        # 4. Save
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

if __name__ == "__main__":
    main()
