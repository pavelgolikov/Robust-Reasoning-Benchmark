
import argparse
import os
import sys
import json
import time
import random
import re
import io
import contextlib
import traceback
import signal
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Fix path to include 'analysis' so 'variables' can be imported by util if needed
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'analysis'))

from datasets import load_dataset
from util import get_prompts, remove_latex_comments, extract_and_grade
from api_utils import generate_response

# ======================================================
# PART 1: HELPER FUNCTIONS (Parser & Executor) - COPIED FROM evaluate_agent.py
# ======================================================

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution Timed Out")

def extract_python_code(response_text):
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def execute_python_code(code_str, state_dict, stdin_input, timeout_sec=10):
    output_capture = io.StringIO()
    signal.signal(signal.SIGALRM, timeout_handler)
    
    original_stdin = sys.stdin
    stdin_mock = io.StringIO(stdin_input)
    sys.stdin = stdin_mock
    
    try:
        with contextlib.redirect_stdout(output_capture):
            signal.alarm(timeout_sec)
            try:
                exec(code_str, state_dict)
            finally:
                signal.alarm(0)
        
        result = output_capture.getvalue()
        if not result:
            return "[Code ran successfully, but produced no output. Make sure to NOT use stdin or input() calls.]"
        return result
    
    except TimeoutException:
         partial = output_capture.getvalue()
         return f"{partial}\n\n--- EXECUTION TIMED OUT ({timeout_sec}s) ---"
    
    except Exception:
        partial_output = output_capture.getvalue()
        error_trace = traceback.format_exc()
        return f"{partial_output}\n\n--- EXECUTION ERROR ---\n{error_trace}"
    
    finally:
        sys.stdin = original_stdin

# ======================================================
# PART 2: AGENT STATE
# ======================================================

@dataclass
class AgentState:
    id: str
    problem_id: str
    sample_idx: int
    experiment_name: str
    
    original_problem: str
    ground_truth: str
    system_prompt_static: str
    user_prompt_content: str
    
    history: List[Dict[str, str]]
    memory: Dict[str, Any] = field(default_factory=dict)
    
    is_done: bool = False
    final_output: str = ""
    extracted_answer: str = ""
    is_correct: bool = False
    step_count: int = 0
    max_steps: int = 20

# ======================================================
# PART 3: AGENT LOOP
# ======================================================

def run_agent_loop(agent: AgentState, model_name: str, max_tokens: int, provider: str = None):
    while not agent.is_done and agent.step_count < agent.max_steps:
        print(f"  [Agent {agent.id}] Step {agent.step_count+1}...")
        
        try:
            # Generate response
            response_text = generate_response(agent.history, model_name, provider=provider, max_tokens=max_tokens)
            agent.step_count += 1
            
            # Append assistant message
            agent.history.append({"role": "assistant", "content": response_text})
            
            # Check for code
            code_block = extract_python_code(response_text)
            
            if code_block:
                try:
                    stdin_content = agent.user_prompt_content.split("TRANSFORMED INPUT:", 1)[1].strip()
                except IndexError:
                    stdin_content = "" # Should fallback gracefully matching protocol
                
                execution_result = execute_python_code(code_block, agent.memory, stdin_content)
                tool_msg = f"Observation:\n{execution_result}"
                agent.history.append({"role": "user", "content": tool_msg})
                
            else:
                # No code, check if done (boxed) or just text
                if "\\boxed{" in response_text:
                    agent.final_output = response_text
                    agent.is_done = True
                elif agent.step_count >= agent.max_steps:
                    agent.final_output = response_text
                    agent.is_done = True
                
        except Exception as e:
            print(f"  [Agent {agent.id}] Error: {e}")
            agent.final_output = f"ERROR: {e}"
            agent.is_done = True

    # Post-process using Math-Verify
    extracted, is_correct = extract_and_grade(agent.final_output, agent.ground_truth)
    agent.extracted_answer = extracted
    agent.is_correct = is_correct

# ======================================================
# PART 4: MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate manual agent implementation (API Version)")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash", help="Name of the API model")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="HuggingFace dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of experiment names")
    parser.add_argument("--num_distractors", type=int, default=32, help="Number of distractors for split_indices")
    parser.add_argument("--provider", type=str, default=None, help="API Provider (google, openai, anthropic). Optional.")
    parser.add_argument("--max_tokens", type=int, required=True, help="Max output tokens (required).")

    args = parser.parse_args()

    if args.names == 'all':
        experiment_names = [ 'context_saturation', 'interleaved_context_line', 'interleaved_context_word', 'interleaved_context_symbol',
        'not_not', 'opposites', 'sentence_reversal', 'word_reversal', 'wrappers', 'split_reversal',
        'rail_fence' ]
    else:
        experiment_names = [n.strip() for n in args.names.split(',') if n.strip()]

    print(f"Running experiments: {experiment_names}")

    # Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Load Variables if needed
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

    for exp_name in experiment_names:
        print(f"\nEvaluating Experiment: {exp_name}")
        
        results = []
        stats = {"correct": 0, "total": 0, "failures": 0}
        
        # Prepare Agents
        agents = []
        random.seed(args.seed)
        
        for i, example in enumerate(dataset):
            prob_id = str(example.get('id', i))
            
            extra_context = None
            if exp_name in ['interleaved_context_word', 'interleaved_context_line', 'interleaved_context_symbol']:
                next_idx = (i + 1) % len(dataset)
                extra_context = remove_latex_comments(dataset[next_idx]['problem'])
            
            current_vars = extracted_vars.get(prob_id) if extracted_vars else None
            cleaned_problem = remove_latex_comments(example['problem'])
            
            for sample_idx in range(args.n_samples):
                current_seed = args.seed + sample_idx + (i * 1000)
                try:
                    final_user_prompt, system_prompt = get_prompts(
                        cleaned_problem, 
                        exp_name, 
                        extra_context, 
                        variables=current_vars,
                        seed=current_seed,
                        num_distractors=args.num_distractors,
                        agentic=True
                    )
                    
                    agent = AgentState(
                        id=f"{prob_id}_sample_{sample_idx}",
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
                    print(f"Skipping {prob_id} (Sample {sample_idx}): {e}")

        # Run Agents Sequentially
        print(f"Running {len(agents)} agents for {exp_name}...")
        for j, agent in enumerate(agents):
            print(f"Agent {j+1}/{len(agents)} ({agent.id}) running...")
            run_agent_loop(agent, args.model, max_tokens=args.max_tokens, provider=args.provider)
            
            # Collect result
            res = {
                "id": agent.problem_id,
                "sample_idx": agent.sample_idx,
                "extracted": agent.extracted_answer,
                "correct": agent.is_correct,
                "original_problem": agent.original_problem,
                "ground_truth": agent.ground_truth,
                "history_dump": [h['content'] for h in agent.history] 
            }
            results.append(res)
            
            stats["total"] += 1
            if agent.is_correct:
                stats["correct"] += 1
            if agent.extracted_answer is None:
                stats["failures"] += 1

        # Save Results
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"Results for {exp_name}: Accuracy {acc:.2%} ({stats['correct']}/{stats['total']})")
        results.append({
            "summary": {
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"],
                "failures": stats["failures"]
            }
        })
        
        experiment_dir = os.path.join(base_dir, exp_name)
        final_output_dir = os.path.join(experiment_dir, "results_agent", safe_model_name, safe_dataset_name)
        os.makedirs(final_output_dir, exist_ok=True)
        
        run_id = f"{safe_model_name}_{safe_dataset_name}_{exp_name}_s{args.seed}_{timestamp}_MANUAL_AGENT"
        json_file = os.path.join(final_output_dir, f"{run_id}.json")
        
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {json_file}")

if __name__ == "__main__":
    main()
