import argparse
import os
import sys
import json
import time
import random
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datasets import load_dataset
import torch
# from vllm import LLM, SamplingParams (Moved to main)

# Ensure we can import from experiments.
# evaluate_conversation.py is in experiments/
# util.py is in experiments/
# context_saturation/generate_systems.py is in experiments/context_saturation/

# Add project root to path just in case
script_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = script_dir
project_dir = os.path.dirname(experiments_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Helper imports
from experiments.util import (
    extract_answer, 
    normalize_answer, 
    remove_latex_comments, 
    BASELINE_SYSTEM_PROMPT
)
from experiments.context_saturation.generate_systems import generate_system

@dataclass
class AgentState:
    id: str
    problem_id: str
    sample_idx: int
    
    # Static Data
    original_problem: str
    ground_truth: str
    variables: List[str] # For distractor generation
    target_distractors: int
    
    # Dynamic State
    history: List[Dict[str, str]]
    current_sys_index: int = 0
    phase: str = "FEEDING" # "FEEDING", "SOLVING", "DONE"
    
    # Output
    is_done: bool = False
    final_output: str = ""
    extracted_answer: str = ""
    is_correct: bool = False
    step_count: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)
    last_distractor_count: int = 0 # Tracks how many distractors were sent in the pending turn
    
    def get_vllm_prompt(self, tokenizer):
        return tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)


def run_single_turn(active_agents: List[AgentState], llm, tokenizer, sampling_params, distractors_per_query, seed):
    """
    Runs a single turn (prompt generation -> vllm generate -> state update) for all active agents.
    """
    current_batch_agents = []
    prompts = []
    
    # 1. Update Prompts based on Phase
    for agent in active_agents:
        if agent.is_done: continue
        
        # --- PHASE LOGIC ---
        if agent.phase == "FEEDING":
            # Check if we are done feeding
            if agent.current_sys_index >= agent.target_distractors:
                agent.phase = "SOLVING"
            else:
                vars_len = len(agent.variables)
                
                # Calculate how many to send
                remaining = agent.target_distractors - agent.current_sys_index
                count = min(distractors_per_query, remaining)
                
                prompt_parts = []
                prompt_parts.append(f"Solve these {count} math problems and output your solutions numbered.\n")
                
                for k in range(count):
                    idx = agent.current_sys_index + k
                    cur_term_ind = (idx * 2) % vars_len
                    # DETERMINISTIC SEEDING: Ensure distractor K for Sample N is always same
                    local_seed = seed + agent.sample_idx * 100000 + idx
                    random.seed(local_seed)
                    
                    distractor_text = generate_system(agent.variables, cur_term_ind, idx).strip()
                    prompt_parts.append(f"{k+1}. {distractor_text}")
                    
                full_prompt = "\n".join(prompt_parts)
                agent.history.append({"role": "user", "content": full_prompt})
                agent.last_distractor_count = count
        
        if agent.phase == "SOLVING":
            # Only add if the last message was NOT user (i.e., we are ready for new input).
            last_role = agent.history[-1]["role"]
            if last_role == "assistant" or last_role == "system":
                # Add Real Problem
                real_problem = remove_latex_comments(agent.original_problem)
                agent.history.append({"role": "user", "content": real_problem})
        
        # Prepare VLLM prompt
        try:
            prompt = agent.get_vllm_prompt(tokenizer)
            prompts.append(prompt)
            current_batch_agents.append(agent)
        except Exception as e:
            print(f"[Error] Failed to format prompt for agent {agent.id}: {e}")
            agent.final_output = f"ERROR_PROMPT_FORMAT: {e}"
            agent.is_done = True
    
    if not current_batch_agents:
        return []
        
    print(f"[Turn Execution] Generating for {len(current_batch_agents)} agents...")
    
    # 2. Generate
    # vLLM handles batching internally for the list of prompts
    try:
        outputs = llm.generate(prompts, sampling_params)
        
        for agent, out_obj in zip(current_batch_agents, outputs):
            if not out_obj.outputs:
                agent.final_output = "ERROR: No output"
                agent.is_done = True
                continue
            
            response_text = out_obj.outputs[0].text
            agent.history.append({"role": "assistant", "content": response_text})
            agent.step_count += 1
            
            # Token Tracking
            if hasattr(out_obj.outputs[0], 'token_ids'):
                token_count = len(out_obj.outputs[0].token_ids)
            else:
                token_count = len(response_text.split()) 
            
            # --- POST-GENERATION UPDATE ---
            if agent.phase == "FEEDING":
                start_idx = agent.current_sys_index
                end_idx = start_idx + agent.last_distractor_count - 1
                
                agent.token_usage[f"distractors {start_idx}-{end_idx}"] = token_count
                
                # Increment index by the number of items we sent
                agent.current_sys_index += agent.last_distractor_count
                # Loop will handle transition to SOLVING next time.
                
            elif agent.phase == "SOLVING":
                # We just got a reply to the real problem.
                agent.token_usage["solution"] = token_count
                
                # This is the final answer.
                agent.final_output = response_text
                agent.is_done = True
                
    except BaseException as e:
        print(f"[Batch Error] {e}")
        # Fail all in batch for safety
        for agent in current_batch_agents:
            agent.is_done = True
            agent.final_output = f"CRITICAL_BATCH_ERROR: {e}"

    return current_batch_agents


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Turn Conversation Agent (Context Saturation)")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_range", type=str, default=None, help="Range of sample indices to process, e.g. '0-10' or '5' or '1,3,5'")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--max_model_length", type=int, default=65536)
    parser.add_argument("--num_distractors", type=int, default=32, help="Number of distractors (conversation turns) before the real problem.")
    parser.add_argument("--distractors_per_query", type=int, default=1, help="Number of distractors to batch in a single user turn.")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to process in this run (for testing/time-limits).")
    parser.add_argument("--dry", action="store_true", help="Run without loading model (fake outputs)")
    parser.add_argument("--new_run", action="store_true", help="Force a new run (ignore existing checkpoints)")
    
    args = parser.parse_args()
    
    # 1. Load Extracted Variables
    extracted_vars = {}
    vars_path = os.path.join(experiments_dir, 'analysis', 'variables', 'extracted_terms_by_problem.json')
    if os.path.exists(vars_path):
        with open(vars_path, 'r') as f:
            extracted_vars = json.load(f)
        # Normalize
        for k, v in extracted_vars.items():
            extracted_vars[k] = [x.replace(" ", "_") for x in v]
        print(f"Loaded variables for {len(extracted_vars)} problems.")
    else:
        print("Warning: Variable file not found. Using default variables.")
        
    # 2. Init VLLM or Mock
    if args.dry:
        print("DRY RUN: Using Mock LLM.")
        class MockOutput:
            def __init__(self, text): 
                self.text = text
                # Mock token_ids
                self.token_ids = [0] * len(text.split())
        class MockCompletion:
            def __init__(self, text): self.outputs = [MockOutput(text)]
        class MockTokenizer:
            def apply_chat_template(self, history, tokenize=False, add_generation_prompt=True):
                return json.dumps(history) # Just dump history as string
        class MockLLM:
            def generate(self, prompts, params):
                return [MockCompletion("Fake Model Output") for _ in prompts]
            def get_tokenizer(self): return MockTokenizer()
            
        llm = MockLLM()
        tokenizer = llm.get_tokenizer()
        sampling_params = None
    else:
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
            tokenizer = llm.get_tokenizer()
            sampling_params = SamplingParams( temperature=0.6, max_tokens=args.max_model_length)
        except Exception as e:
            print(f"Failed to initialize vLLM: {e}")
            exit(1)
        
    # 3. Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    
    # Process sample_range
    if args.sample_range:
        indices = []
        parts = args.sample_range.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end)) # Python range exclusive
            else:
                indices.append(int(part))
        # Validate indices
        indices = [i for i in indices if 0 <= i < len(dataset)]
        if not indices:
             print("Error: No valid indices found in sample_range.")
             exit(1)
        print(f"Selecting {len(indices)} samples based on range: {args.sample_range}")
        dataset = dataset.select(indices)

    # We now iterate over the *current* dataset indices (0 to len(dataset)-1)
    # forcing a list to act as our 'indices' to chunk
    indices = list(range(len(dataset)))

    default_vars = ["x", "y", "n", "k", "A", "B", "S"]
    random.seed(args.seed)
    
    results = []
    stats = {"correct": 0, "total": 0, "failures": 0}

    # --- RESUME LOGIC ---
    # Construct expected directory
    safe_model_name = args.model.replace('/', '_').replace(' ', '_')
    safe_dataset_name = args.dataset.replace('/', '_')
    res_dir = os.path.join(experiments_dir, "context_saturation", "conv_results", safe_model_name, safe_dataset_name)
    os.makedirs(res_dir, exist_ok=True)
    
    # Check for latest file matching pattern
    import glob
    existing_files = glob.glob(os.path.join(res_dir, f"{safe_model_name}_{safe_dataset_name}_s{args.seed}_*_CONVERSATION.json"))
    existing_files.sort(key=os.path.getmtime) # Sort by time, latest last

    json_file = None
    existing_results = []
    
    if existing_files and not args.new_run:
        latest_file = existing_files[-1]
        print(f"[Resume] Found existing run: {latest_file}")
        try:
            with open(latest_file, "r") as f:
                existing_results = json.load(f)
            
            # Map existing sample_indices to skip
            finished_keys = set((item.get('id'), item.get('sample_idx')) for item in existing_results)
            print(f"[Resume] Loaded {len(existing_results)} completed samples. Filtering queue...")
            
            print(f"[Resume] Will skip {len(finished_keys)} already completed items.")
            
            json_file = latest_file
            results = existing_results # Pre-populate current results list
            
            # Recalculate stats
            stats["total"] = len(existing_results)
            stats["correct"] = sum(1 for r in existing_results if r.get("correct"))
            stats["failures"] = sum(1 for r in existing_results if not r.get("correct"))
            
        except Exception as e:
             print(f"[Resume] Error reading file {latest_file}, starting fresh. Error: {e}")

    if not json_file:
        # Start New
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(res_dir, f"{safe_model_name}_{safe_dataset_name}_s{args.seed}_{timestamp}_CONVERSATION.json")
        print(f"[Init] Starting new run. Output: {json_file}")
    
    # --------------------
    
    # results = [] (Removed, initialized above)
    # stats = ... (Removed, initialized above)
    # timestamp = ... (Removed)
    
    # 4. Initialize ALL Agents
    print(f"Initializing agents for {len(indices)} samples...")
    all_agents = []
    
    for idx in indices:
        example = dataset[idx]
        prob_id = str(example.get('id', idx))
        
        # Get variables
        current_vars = extracted_vars.get(prob_id, default_vars)
        if len(current_vars) < 2: 
            current_vars = default_vars
            
        for sample_idx in range(args.n_samples):
            # Create Agent
            agent = AgentState(
                id=f"{prob_id}_s{sample_idx}",
                problem_id=prob_id,
                sample_idx=sample_idx,
                original_problem=example['problem'],
                ground_truth=example['answer'],
                variables=current_vars,
                target_distractors=args.num_distractors,
                history=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT}
                ]
            )
            all_agents.append(agent)

    # RESUME FILTERING for all agents
    if existing_results:
         active_agents = [a for a in all_agents if (a.problem_id, a.sample_idx) not in finished_keys]
         print(f"[Resume] Filtered down to {len(active_agents)} active agents (from {len(all_agents)} total).")
    else:
         active_agents = all_agents

    if not active_agents:
        print("All agents completed. Exiting.")
        exit(0)

    print(f"Starting parallel execution for {len(active_agents)} agents...")
    
    # 5. Main Execution Loop
    step_num = 0
    while active_agents:
        step_num += 1
        print(f"\n[Global Turn {step_num}] Processing...")
        
        # Run one turn
        try:
            # We pass ALL active agents. vLLM handles batching internally.
            processed_agents = run_single_turn(active_agents, llm, tokenizer, sampling_params, args.distractors_per_query, args.seed)
        except torch.cuda.OutOfMemoryError:
            print(f"[CRITICAL] CUDA OOM Error on Turn {step_num}!")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            # If OOM, we might need to fail everyone or retry?
            # For now, let's just fail everyone to be safe and save what we can.
            for agent in active_agents:
                if not agent.is_done:
                    agent.final_output = "ERROR_CUDA_OOM"
                    agent.is_done = True
            
        # Check output & Save Progress
        newly_finished = []
        still_active = []
        
        for agent in active_agents:
            if agent.is_done:
                newly_finished.append(agent)
            else:
                still_active.append(agent)
                
        if newly_finished:
            print(f"Turn {step_num}: {len(newly_finished)} agents finished. Saving...")
            
            for agent in newly_finished:
                extracted = extract_answer(agent.final_output)
                is_correct = normalize_answer(extracted) == normalize_answer(agent.ground_truth)
                
                agent.is_correct = is_correct
                agent.extracted_answer = extracted
                
                stats["total"] += 1
                if is_correct: stats["correct"] += 1
                else: stats["failures"] += 1
                
                # Aggregate Token Usage
                distractor_tokens = [v for k, v in agent.token_usage.items() if k.startswith("distractor")]
                solution_tokens = agent.token_usage.get("solution", 0)
                
                token_usage_summary = {
                    "distractors_total_tokens": sum(distractor_tokens),
                    "distractors_avg_tokens": sum(distractor_tokens) / len(distractor_tokens) if distractor_tokens else 0,
                    "distractors_count": len(distractor_tokens),
                    "solution_tokens": solution_tokens
                }

                results.append({
                    "id": agent.problem_id,
                    "sample_idx": agent.sample_idx,
                    "system_prompt": BASELINE_SYSTEM_PROMPT,
                    "output": agent.final_output,
                    "extracted": extracted,
                    "correct": is_correct,
                    "token_usage": token_usage_summary,
                    "original_problem": agent.original_problem,
                    "ground_truth": agent.ground_truth,
                    "history_dump": [h['content'] for h in agent.history]
                })

            # Save File
            try:
                with open(json_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"[Progressive Save] Saved {len(results)} samples (Total) to: {json_file}")
            except Exception as e:
                print(f"[Warning] Failed to save progressive results: {e}")

        active_agents = still_active
        
        # GC to keep memory clean
        if step_num % 5 == 0:
            import gc
            gc.collect()

    # 5. Final Report
    # (Results are already saved, just print stats)
        
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"Results: Accuracy {acc:.2%} ({stats['correct']}/{stats['total']})")
    print(f"Final Results saved to: {json_file}")

if __name__ == "__main__":
    main()
