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
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list) # Stores probes
    
    def get_vllm_prompt(self, tokenizer):
        return tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)


def run_single_turn(active_agents: List[AgentState], llm, tokenizer, sampling_params, distractors_per_query, seed):
    """
    Runs a single turn (Prob + Feed) for all active agents.
    1. PROBE: Check accuracy on Real Problem (without modifying history).
    2. FEED: Generate Distractor Solutions (modifying history).
    Handles Context Filling Strategy with Backtracking.
    """
    batch_meta = [] # Stores metadata for each prompt: {agent, type='PROBE'|'FEED'|'SOLVE'}
    prompts = []
    
    # 1. Prepare Prompts
    for agent in active_agents:
        if agent.is_done: continue
        
        # --- PHASE LOGIC ---
        # --- PHASE: FEEDING ---
        if agent.phase == "FEEDING":
            
            # PREPARE FEED (Context Filling)
            # Context Manager Check removed

            # Standard Distractor Generation
            vars_len = len(agent.variables)
            remaining = agent.target_distractors - agent.current_sys_index
            count = min(distractors_per_query, remaining)
            
            if count <= 0:
                 agent.phase = "SOLVING"
            else:
                prompt_parts = []
                prompt_parts.append(f"Solve these {count} math problems and output your solutions numbered.\n")
                
                for k in range(count):
                    idx = agent.current_sys_index + k
                    cur_term_ind = (idx * 2) % vars_len
                    local_seed = seed + agent.sample_idx * 100000 + idx
                    random.seed(local_seed)
                    
                    distractor_text = generate_system(agent.variables, cur_term_ind, idx).strip()
                    prompt_parts.append(f"{k+1}. {distractor_text}")
                    
                full_prompt = "\n".join(prompt_parts)
                # Modify history for FEED
                agent.history.append({"role": "user", "content": full_prompt})
                agent.last_distractor_count = count
                
                try:
                    feed_prompt = agent.get_vllm_prompt(tokenizer)
                    prompts.append(feed_prompt)
                    batch_meta.append({"agent": agent, "type": "FEED"})
                except Exception as e:
                    print(f"[Feed Error] Agent {agent.id}: {e}")
                    agent.final_output = f"ERROR_PROMPT_FORMAT: {e}"
                    agent.is_done = True
        
        # --- PHASE: SOLVING (Legacy/Final Turn) ---
        elif agent.phase == "SOLVING":
            # Add Real Problem
            last_role = agent.history[-1]["role"]
            if last_role == "assistant" or last_role == "system":
                real_problem = remove_latex_comments(agent.original_problem)
                agent.history.append({"role": "user", "content": "Solve the following question using regular mathematics.\n\n" + real_problem})
            
            try:
                prompt = agent.get_vllm_prompt(tokenizer)
                prompts.append(prompt)
                batch_meta.append({"agent": agent, "type": "SOLVE"})
            except Exception as e:
                agent.is_done = True
                print(f"[Solve Error] {e}")

    if not prompts:
        return []
        
    print(f"[Turn Execution] Generating {len(prompts)} items ({len(active_agents)} agents)...")
    
    # 2. Generate
    try:
        outputs = llm.generate(prompts, sampling_params)
        
        # Process Outputs matching metadata
        for meta, out_obj in zip(batch_meta, outputs):
            agent = meta["agent"]
            generation_type = meta["type"]
            
            if not out_obj.outputs:
                if generation_type != "PROBE":
                    agent.final_output = "ERROR: No output"
                    agent.is_done = True
                continue
            
            response_text = out_obj.outputs[0].text

            # Valid Output (FEED/SOLVE) - append to history
            agent.history.append({"role": "assistant", "content": response_text})
            agent.step_count += 1
            
            # Token Tracking (Standard)
            if hasattr(out_obj.outputs[0], 'token_ids'):
                token_count = len(out_obj.outputs[0].token_ids)
            else:
                token_count = len(response_text.split()) 
            
            if agent.phase == "FEEDING":
                start_idx = agent.current_sys_index
                end_idx = start_idx + agent.last_distractor_count - 1
                agent.token_usage[f"distractors {start_idx}-{end_idx}"] = token_count
                
                # Increment index by the number of items we sent
                agent.current_sys_index += agent.last_distractor_count
                
            elif agent.phase == "SOLVING":
                agent.token_usage["solution"] = token_count
                agent.final_output = response_text
                agent.is_done = True
                
    except BaseException as e:
        print(f"[Batch Error] {e}")
        # Fail all is safest, but we could try to be granular.
        for meta in batch_meta:
            agent = meta["agent"]
            if not agent.is_done:
                agent.final_output = f"CRITICAL_BATCH_ERROR: {e}"
                agent.is_done = True

    return active_agents

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Turn Conversation Agent (Context Saturation)")
    # parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # parser.add_argument("--quantization", type=str, default=None, help="Quantization mode (e.g., 'fp8', 'awq', 'gptq')")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_range", type=str, default=None, help="Range of sample indices to process, e.g. '0-10' or '5' or '1,3,5'")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_model_length", type=int, default=65536)
    parser.add_argument("--num_distractors", type=int, default=8, help="Number of distractors (conversation turns) before the real problem.")
    parser.add_argument("--distractors_per_query", type=int, default=1, help="Number of distractors to batch in a single user turn.")
    parser.add_argument("--dry", action="store_true", help="Run without loading model (fake outputs)")
    parser.add_argument("--context_pollution_percent", type=int, default=None, help="Target context pollution saturation in percent (0-100). Overrides num_distractors.")
    
    args = parser.parse_args()

    # --- Logic: Context Pollution Percent Calculation ---
    EST_TOKENS_PER_DISTRACTOR = 2048
    if args.context_pollution_percent is not None:
        if not (0 < args.context_pollution_percent <= 100):
             print(f"Error: Context pollution percent must be between 0 and 100. Got {args.context_pollution_percent}")
             exit(1)
             
        target_tokens = args.max_model_length * (args.context_pollution_percent / 100.0)
        calculated_distractors = target_tokens / EST_TOKENS_PER_DISTRACTOR
        
        # User requested: Round to nearest whole number, no error.
        num_distractors = int(round(calculated_distractors))
        
        if num_distractors % args.distractors_per_query != 0:
             remaining = num_distractors % args.distractors_per_query
             print(f"[Context Pollution] Warning: Calculated distractors ({num_distractors}) is not divisible by distractors_per_query ({args.distractors_per_query}).")
             print(f"[Context Pollution] The final batch will contain only {remaining} distractor(s).")
             # Loop logic at lines ~104 handles min(batch, remaining) automatically.
             
        print(f"[Context Pollution] Override: {args.context_pollution_percent}% of {args.max_model_length} = {target_tokens:.0f} tokens.")
        print(f"[Context Pollution] Setting num_distractors = {num_distractors} (was {args.num_distractors})")
        args.num_distractors = num_distractors
    # ----------------------------------------------------
    
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
            tokens_per_distractor_response = 3000
            token_limit = args.distractors_per_query * tokens_per_distractor_response
            sampling_params = SamplingParams(
                temperature=0.6,
                max_tokens=min(token_limit, args.max_model_length),
                repetition_penalty=1.1
            )
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
    indices = list(range(len(dataset)))

    default_vars = ["x", "y", "n", "k", "A", "B", "S"]
    random.seed(args.seed)
    
    # Output Setup (Always New)
    safe_model_name = args.model.replace('/', '_').replace(' ', '_')
    safe_dataset_name = args.dataset.replace('/', '_')
    res_dir = os.path.join(experiments_dir, "context_saturation", "conv_results", safe_model_name, safe_dataset_name)
    os.makedirs(res_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(res_dir, f"{safe_model_name}_{safe_dataset_name}_s{args.seed}_{timestamp}_CONVERSATION.json")
    print(f"[Init] Starting new run. Output: {json_file}")
    
    results_dict = {}
    stats = {"correct": 0, "total": 0, "failures": 0}
    
    # 4. Initialize ALL Agents
    print(f"Initializing agents for {len(indices)} samples...")
    active_agents = []
    
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
            active_agents.append(agent)
            
            # Initial placeholder in results
            results_dict[agent.id] = {
                 "id": agent.problem_id,
                 "sample_idx": agent.sample_idx,
                 "is_done": False,
                 "metadata": {"phase": "INIT"}
            }

    print(f"Starting execution for {len(active_agents)} agents...")

    # 5. Main Execution Loop
    step_num = 0
    while active_agents:
        step_num += 1
        print(f"\n[Global Turn {step_num}] Processing...")
        
        # Run one turn
        try:
            processed_agents = run_single_turn(active_agents, llm, tokenizer, sampling_params, args.distractors_per_query, args.seed)
        except torch.cuda.OutOfMemoryError:
            print(f"[CRITICAL] CUDA OOM Error on Turn {step_num}!")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            # Fail active agents
            for agent in active_agents:
                if not agent.is_done:
                    agent.final_output = "ERROR_CUDA_OOM"
                    agent.is_done = True
            
        # Partition Active Agents
        newly_finished = []
        still_active = []
        
        for agent in active_agents:
            if agent.is_done:
                newly_finished.append(agent)
            else:
                still_active.append(agent)

        # SAVE PROGRESS
        for agent in active_agents:
            # Prepare data
            extracted = extract_answer(agent.final_output)
            is_correct = normalize_answer(extracted) == normalize_answer(agent.ground_truth)
            
            # Aggregate Token Usage
            distractor_tokens = [v for k, v in agent.token_usage.items() if k.startswith("distractor")]
            solution_tokens = agent.token_usage.get("solution", 0)
            
            token_usage_summary = {
                "distractors_total_tokens": sum(distractor_tokens),
                "distractors_avg_tokens": sum(distractor_tokens) / len(distractor_tokens) if distractor_tokens else 0,
                "distractors_count": len(distractor_tokens),
                "solution_tokens": solution_tokens
            }
            
            metadata = {
                "current_sys_index": agent.current_sys_index,
                "phase": agent.phase,
                "token_usage": agent.token_usage
            }

            entry = {
                "id": agent.problem_id,
                "sample_idx": agent.sample_idx,
                "system_prompt": BASELINE_SYSTEM_PROMPT,
                "output": agent.final_output,
                "extracted": extracted,
                "correct": is_correct,
                "is_done": agent.is_done,
                "metadata": metadata,
                "token_usage": token_usage_summary,
                "original_problem": agent.original_problem,
                "ground_truth": agent.ground_truth,
                "history_dump": [h['content'] for h in agent.history],
                "intermediate_results": agent.intermediate_results
            }
            
            results_dict[agent.id] = entry

        if newly_finished:
            print(f"Turn {step_num}: {len(newly_finished)} agents finished.")
            for agent in newly_finished:
                 extracted = extract_answer(agent.final_output)
                 is_correct = normalize_answer(extracted) == normalize_answer(agent.ground_truth)
                 stats["total"] += 1
                 if is_correct: stats["correct"] += 1
                 else: stats["failures"] += 1

        try:
            with open(json_file, "w") as f:
                json.dump(list(results_dict.values()), f, indent=2)
            print(f"[Save] Updated results file.")
        except Exception as e:
            print(f"[Warning] Failed to save results: {e}")
            
        active_agents = still_active
        
        # GC
        if step_num % 5 == 0:
            import gc
            gc.collect()

    # 5. Final Report
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"Results: Accuracy {acc:.2%} ({stats['correct']}/{stats['total']})")
    print(f"Final Results saved to: {json_file}")

if __name__ == "__main__":
    main()
