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
    
    def get_vllm_prompt(self, tokenizer):
        return tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)


def run_batch_execution(agents: List[AgentState], llm, tokenizer, sampling_params):
    """
    Runs the multi-turn batched execution loop.
    """
    active_agents = [a for a in agents if not a.is_done]
    
    while active_agents:
        # 1. Update Prompts based on Phase
        current_batch_agents = []
        prompts = []
        
        for agent in active_agents:
            if agent.is_done: continue
            
            # --- PHASE LOGIC ---
            if agent.phase == "FEEDING":
                # Check if we are done feeding
                if agent.current_sys_index >= agent.target_distractors:
                    agent.phase = "SOLVING"
                else:
                    vars_len = len(agent.variables)
                    cur_term_ind = (agent.current_sys_index * 2) % vars_len
                    
                    distractor_text = generate_system(agent.variables, cur_term_ind, agent.current_sys_index)
                    distractor_text = distractor_text.strip()
                    
                    agent.history.append({"role": "user", "content": distractor_text})
            
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
            break
            
        print(f"\n[Batch Step] Generating for {len(current_batch_agents)} agents...")
        
        # 2. Generate
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
                    # Fallback for Mock or if token_ids missing
                    token_count = len(response_text.split()) 
                
                # --- POST-GENERATION UPDATE ---
                if agent.phase == "FEEDING":
                    # We just got a reply to a distractor.
                    agent.token_usage[f"distractor {agent.current_sys_index}"] = token_count
                    
                    # Increment index
                    agent.current_sys_index += 1
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

        # Filter active
        active_agents = [a for a in agents if not a.is_done]


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
    parser.add_argument("--batch_size", type=int, default=10, help="Number of samples to process in parallel (batch size).")
    parser.add_argument("--dry", action="store_true", help="Run without loading model (fake outputs)")
    
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

    # 4. Process in Batches
    print(f"Starting execution with batch size: {args.batch_size}")
    
    indices = list(range(len(dataset)))

    default_vars = ["x", "y", "n", "k", "A", "B", "S"]
    random.seed(args.seed)
    
    results = []
    stats = {"correct": 0, "total": 0, "failures": 0}
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Chunk indices
    for i in range(0, len(indices), args.batch_size):
        batch_indices = indices[i : i + args.batch_size]
        print(f"\n[Main Loop] Processing batch {i//args.batch_size + 1} ({len(batch_indices)} samples)...")
        
        batch_agents = []
        
        for idx in batch_indices:
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
                batch_agents.append(agent)
                
        if not batch_agents:
            continue
            
        print(f"Initialized {len(batch_agents)} agents for this batch. Running execution...")
        
        # Run Batch with OOM Handling
        try:
            run_batch_execution(batch_agents, llm, tokenizer, sampling_params)
        except torch.cuda.OutOfMemoryError:
            print(f"[CRITICAL] CUDA OOM Error on batch {i//args.batch_size + 1}!")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            for agent in batch_agents:
                if not agent.is_done:
                    agent.final_output = "ERROR_CUDA_OOM"
                    agent.is_done = True
        except Exception as e:
             print(f"[CRITICAL] Unknown Error on batch {i//args.batch_size + 1}: {e}")
             for agent in batch_agents:
                if not agent.is_done:
                    agent.final_output = f"ERROR_UNKNOWN: {e}"
                    agent.is_done = True

        
        # Collect Results
        for agent in batch_agents:
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
            
        # Cleanup
        del batch_agents
        import gc
        gc.collect()

        # --- PROGRESSIVE SAVING ---
        # Save results after EACH batch to prevent data loss safely (overwriting file)
        safe_model_name = args.model.replace('/', '_').replace(' ', '_')
        safe_dataset_name = args.dataset.replace('/', '_')
        res_dir = os.path.join(experiments_dir, "context_saturation", "conv_results", safe_model_name, safe_dataset_name)
        os.makedirs(res_dir, exist_ok=True)
        json_file = os.path.join(res_dir, f"{safe_model_name}_{safe_dataset_name}_s{args.seed}_{timestamp}_CONVERSATION.json")
        
        try:
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[Progressive Save] Saved {len(results)} samples to: {json_file}")
        except Exception as e:
            print(f"[Warning] Failed to save progressive results: {e}")

    # 5. Final Report
    # (Results are already saved, just print stats)
        
    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"Results: Accuracy {acc:.2%} ({stats['correct']}/{stats['total']})")
    print(f"Final Results saved to: {json_file}")

if __name__ == "__main__":
    main()
