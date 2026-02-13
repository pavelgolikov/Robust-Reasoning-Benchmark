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

TOK_PER_DISTRACTOR = 2048

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
    variables: List[str] # For distractor generation
    
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
    context_token_count: int = 0 # Continuously tracks total context tokens (User + Assistant)
    sampling_params: Dict[str, Any] = field(default_factory=dict) # Tracks params used for Feed and Solve
    
    def get_vllm_prompt(self, tokenizer):
        return tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)


def run_single_turn(active_agents: List[AgentState], llm, tokenizer, sampling_params, distractors_per_query, seed, saturation_limit=None):
    """
    Runs a single turn (Prob + Feed) for all active agents.
    1. PROBE: Check accuracy on Real Problem (without modifying history).
    2. FEED: Generate Distractor Solutions (modifying history).
    Handles Context Filling Strategy with Backtracking.
    """
    batch_meta = [] # Stores metadata for each prompt: {agent, type='PROBE'|'FEED'|'SOLVE'}
    prompts = []
    
    feed_agents = []
    solve_agents = []

    
    # 1. Prepare Prompts
    # SPLIT LOGIC: Group agents by Phase because they need different SamplingParams
    feed_agents = []
    solve_agents = []
    
    # 1. Prepare Prompts and Separate by Phase
    for agent in active_agents:
        if agent.is_done: continue
        
        # --- PHASE: FEEDING ---
        if agent.phase == "FEEDING":
            # Dynamic Logic
            current_len = agent.context_token_count
            if saturation_limit is not None and current_len >= saturation_limit:
                agent.phase = "SOLVING"
                # Fall through to SOLVING logic immediately
            else:
                # Still FEEDING
                count = distractors_per_query
                vars_len = len(agent.variables)
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
                agent.history.append({"role": "user", "content": full_prompt})
                
                # TRACK USER TOKENS
                if hasattr(tokenizer, 'encode'):
                    agent.context_token_count += len(tokenizer.encode(full_prompt, add_special_tokens=False, truncation=False))
                else:
                    agent.context_token_count += len(full_prompt.split())
                    
                agent.last_distractor_count = count
                
                try:
                    feed_prompt = agent.get_vllm_prompt(tokenizer)
                    feed_agents.append({"agent": agent, "prompt": feed_prompt, "type": "FEED"})
                except Exception as e:
                    print(f"[Feed Error] Agent {agent.id}: {e}")
                    agent.final_output = f"ERROR_PROMPT_FORMAT: {e}"
                    agent.is_done = True
                    
        # --- PHASE: SOLVING ---
        # Note: If agent switched to SOLVING above, it falls here (if we used `if` not `elif` after check)
        # But we used `if agent.phase == "FEEDING"` then check saturation.
        # If switched, it is now "SOLVING", so strict `if` below handles it if we re-check or simply continue.
        # Simpler: Check phase again.
        
        if agent.phase == "SOLVING":
             # Add Real Problem
            last_role = agent.history[-1]["role"]
            if last_role == "assistant" or last_role == "system":
                 # START CALCULATION: Context length before real problem
                if agent.context_token_count > 0:
                    pre_solve_tokens = agent.context_token_count
                else:
                    context_str_pre = tokenizer.apply_chat_template(agent.history, tokenize=False, add_generation_prompt=False)
                    if hasattr(tokenizer, 'encode'):
                        pre_solve_tokens = len(tokenizer.encode(context_str_pre, truncation=False))
                    else:
                        pre_solve_tokens = len(context_str_pre.split())
                agent.token_usage["pre_solve_context_tokens"] = pre_solve_tokens
                # END CALCULATION

                real_problem = remove_latex_comments(agent.original_problem)
                
                # Baseline Mode (saturation_limit == 0): No Prefix
                if saturation_limit == 0:
                    agent.history.append({"role": "user", "content": real_problem})
                else:
                    agent.history.append({"role": "user", "content": "Solve the following question using regular mathematics.\n\n" + real_problem})
            
            try:
                prompt = agent.get_vllm_prompt(tokenizer)
                solve_agents.append({"agent": agent, "prompt": prompt, "type": "SOLVE"})
            except Exception as e:
                agent.is_done = True
                print(f"[Solve Error] {e}")

    # 2. EXECUTE BATCHES
    # Determine params
    feed_params = sampling_params["FEED"] if isinstance(sampling_params, dict) else sampling_params
    solve_params = sampling_params["SOLVE"] if isinstance(sampling_params, dict) else sampling_params
    
    # Run FEED Batch
    if feed_agents:
        prompts_feed = [bg["prompt"] for bg in feed_agents]
        print(f"[Turn Execution] Feeding {len(feed_agents)} agents...")
        try:
            outputs_feed = llm.generate(prompts_feed, feed_params)
            process_outputs(feed_agents, outputs_feed, tokenizer, saturation_limit)
        except BaseException as e:
            print(f"[Feed Batch Error] {e}")
            for bg in feed_agents:
                bg["agent"].is_done = True
                bg["agent"].final_output = f"BATCH_ERROR: {e}"

    # Run SOLVE Batch
    if solve_agents:
        prompts_solve = [bg["prompt"] for bg in solve_agents]
        print(f"[Turn Execution] Solving {len(solve_agents)} agents...")
        try:
            outputs_solve = llm.generate(prompts_solve, solve_params)
            process_outputs(solve_agents, outputs_solve, tokenizer, saturation_limit)
        except BaseException as e:
            print(f"[Solve Batch Error] {e}")
            for bg in solve_agents:
                bg["agent"].is_done = True
                bg["agent"].final_output = f"BATCH_ERROR: {e}"

    return active_agents

def process_outputs(batch_meta, outputs, tokenizer, saturation_limit):
    for meta, out_obj in zip(batch_meta, outputs):
        agent = meta["agent"]
        generation_type = meta["type"]
        
        if not out_obj.outputs:
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
        
        # TRACK ASSISTANT TOKENS
        agent.context_token_count += token_count 
        
        # --- POST-GENERATION TRUNCATION LOGIC (Only for FEED) ---
        if agent.phase == "FEEDING" and saturation_limit is not None and saturation_limit > 0:
                if agent.context_token_count > saturation_limit:
                    excess = agent.context_token_count - saturation_limit
                    
                    # 1. Truncate Assistant Output (Last Item)
                    cut_assistant = min(excess, token_count)
                    
                    if cut_assistant > 0:
                        keep_count = token_count - cut_assistant
                        print(f"[Saturation] Agent {agent.id}: Truncating {cut_assistant} tokens from assistant output to hit limit {saturation_limit}. Kept: {keep_count}")
                        
                        # Perform Truncation on Assistant Output
                        if hasattr(out_obj.outputs[0], 'token_ids'):
                            new_ids = out_obj.outputs[0].token_ids[:keep_count]
                            try:
                                truncated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                            except AttributeError:
                                truncated_text = " ".join(response_text.split()[:keep_count])
                        else:
                            words = response_text.split()
                            truncated_text = " ".join(words[:keep_count])
                        
                        agent.history[-1]["content"] = truncated_text
                        token_count = keep_count
                        agent.context_token_count -= cut_assistant
                        excess -= cut_assistant

                    # 2. If still over limit, Truncate User Prompt (Second to Last Item)
                    if excess > 0:
                        user_content = agent.history[-2]["content"]
                        print(f"[Saturation] Agent {agent.id}: Still over limit by {excess}. Truncating user prompt.")
                        
                        if hasattr(tokenizer, 'encode'):
                            user_ids = tokenizer.encode(user_content, add_special_tokens=False, truncation=False)
                            current_user_len = len(user_ids)
                            keep_user = max(0, current_user_len - excess)
                            final_user_ids = user_ids[:keep_user]
                            try:
                                new_user_text = tokenizer.decode(final_user_ids, skip_special_tokens=True)
                            except AttributeError:
                                new_user_text = user_content[:len(user_content)//2] 
                        else:
                            user_words = user_content.split()
                            keep_user = max(0, len(user_words) - excess)
                            new_user_text = " ".join(user_words[:keep_user])
                            
                        agent.history[-2]["content"] = new_user_text
                        agent.context_token_count -= excess
        
        if agent.phase == "FEEDING":
            start_idx = agent.current_sys_index
            end_idx = start_idx + agent.last_distractor_count - 1
            agent.token_usage[f"distractors {start_idx}-{end_idx}"] = token_count
            agent.current_sys_index += agent.last_distractor_count
            
        elif agent.phase == "SOLVING":
            agent.token_usage["solution"] = token_count
            agent.final_output = response_text
            agent.is_done = True



def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Turn Conversation Agent (Context Saturation)")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--split", type=str, default="all", help="Dataset split to use (train/test/all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_range", type=str, default=None, help="Range of sample indices to process, e.g. '0-10' or '5' or '1,3,5'")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_model_length", type=int, default=65536)
    parser.add_argument("--distractors_per_query", type=int, default=1, help="Number of distractors to batch in a single user turn.")
    parser.add_argument("--max_saturation_step_tokens", type=int, default=4096, help="Max tokens allowed for a single Feeding step.")
    parser.add_argument("--dry", action="store_true", help="Run without loading model (fake outputs)")
    parser.add_argument("--context_saturation", type=int, default=None, help="Target context saturation in percent (0-100). Triggers switch to Real Problem when reached.")
    
    args = parser.parse_args()

    # --- Logic: Context Saturation Limit ---
    if args.context_saturation is None:
         print("Error: --context_saturation is required (0-100).")
         exit(1)

    if not (0 <= args.context_saturation <= 100):
         print(f"Error: Context saturation must be between 0 and 100. Got {args.context_saturation}")
         exit(1)
         
    saturation_limit = int(args.max_model_length * (args.context_saturation / 100.0))
    
    if args.context_saturation == 0:
        print(f"[Context Saturation] Baseline Mode (0%). Distractors will be skipped.")
    else:
        print(f"[Context Saturation] Target: {args.context_saturation}% of {args.max_model_length} = {saturation_limit} tokens.")
        print(f"[Context Saturation] Dynamic filling enabled.")
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
            def encode(self, text, **kwargs):
                return [0] * len(text.split())
            def decode(self, token_ids, **kwargs):
                return " ".join(["word"]*len(token_ids))
        class MockLLM:
            def generate(self, prompts, params):
                return [MockCompletion("Fake Model Output") for _ in prompts]
            def get_tokenizer(self): return MockTokenizer()
            
        llm = MockLLM()
        tokenizer = llm.get_tokenizer()
        llm = MockLLM()
        tokenizer = llm.get_tokenizer()
        # Mock Params
        feed_sampling_params = {"temperature": 0.7, "max_tokens": 100}
        solve_sampling_params = {"temperature": 0.7, "max_tokens": 100}
        sampling_params = {"FEED": feed_sampling_params, "SOLVE": solve_sampling_params}
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
            token_limit = args.distractors_per_query * TOK_PER_DISTRACTOR
            if args.context_saturation == 0:
                print("[Setup] Baseline Mode: Temperature=0.7, Max Tokens=Model Length, No Rep Penalty")
            # Unified Sampling Params Logic based on User Request
            # 1. Temperature = 0.7 always
            # 2. No Repetition Penalty
            # 3. Max Tokens: Defined dynamically. Here we set a default `sampling_params`
            #    However, `run_single_turn` will need to likely manage separate params for FEED vs SOLVE.
            #    Let's create two SamplingParams objects.

            # FEEDING Params
            feed_sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=args.max_saturation_step_tokens
            )


            # SOLVING Params
            solve_sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=args.max_model_length
            )

            # We pass a DICT of params to run_single_turn or handle it there.
            # Let's pass a tuple or dict.
            sampling_params = {"FEED": feed_sampling_params, "SOLVE": solve_sampling_params}

            print(f"[Config] Temp=0.7, Feed Max={args.max_saturation_step_tokens}, Solve Max={args.max_model_length}")

        except Exception as e:
            print(f"Failed to initialize vLLM: {e}")
            exit(1)
        
    # 3. Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    if args.split == "train":
        dataset = load_dataset(args.dataset, split="train")
    elif args.split == "test":
        dataset = load_dataset(args.dataset, split="test")
    elif args.split == "all":
        dataset0 = load_dataset(args.dataset, split="train")
        dataset1 = load_dataset(args.dataset, split="test")
        dataset = dataset0.concatenate(dataset1)
    print(f"Dataset loaded with {len(dataset)} samples.")
    
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
                history=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT}
                ],
                context_token_count=len(tokenizer.encode(BASELINE_SYSTEM_PROMPT)) if hasattr(tokenizer, 'encode') else len(BASELINE_SYSTEM_PROMPT.split()),
                sampling_params={
                    "feeding": {
                        "temperature": 0.7,
                        "max_tokens": args.max_saturation_step_tokens
                    },
                    "solving": {
                        "temperature": 0.7,
                        "max_tokens": args.max_model_length
                    }
                }
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
            processed_agents = run_single_turn(active_agents, llm, tokenizer, sampling_params, args.distractors_per_query, args.seed, saturation_limit=saturation_limit)
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
                "solution_tokens": solution_tokens,
                "pre_solve_context_tokens": agent.token_usage.get("pre_solve_context_tokens", 0)
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
                "sampling_params": agent.sampling_params
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
