import argparse
import os
import json
import time
import random
from datasets import load_dataset
from util import get_prompts, remove_latex_comments, BASELINE_SYSTEM_PROMPT, extract_and_grade


# from trim_context import trim_context

FALCON_CHAT_TEMPLATE = """{%- if messages and messages[0]['role'] == 'system' %}
  {% set system_msg = messages[0]['content'] %}  
  {%- set remaining_messages = messages[1:] %}
{%- else %}
  {% set system_msg = "You are Falcon, a helpful AI assistant created by Technology Innovation Institute (TII). To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e., <think> reasoning process here </think> answer here." %}
  {%- set remaining_messages = messages %}
{%- endif %}

{%- if tools %}
<|im_start|>system
{{ system_msg }}
# Tools
You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags.
<tools>
{%- for tool in tools %}
{{- "" }}
{{ tool | tojson }}
{%- endfor %}
{{- "" }}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
<|im_end|>

{%- else %}
<|im_start|>system
{{ system_msg }}
<|im_end|>
{%- endif %}

{# --- Render remaining messages --- #}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}{%- for message in remaining_messages %}
  {%- set content = message.get('content','') %}
  {%- if message['role'] == 'user' %}
    {{- '<|im_start|>' + message['role'] + '\n' + content + '<|im_end|>\n' }}
  {%- elif message['role'] == 'assistant' %}
    {{- '<|im_start|>' + message.role + '\n' }}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- content + '\n' }}
            {%- endif %}
        {%- else %}
            {{- content + '\n' }}
        {%- endif %}
    {%- if tools and message.tool_calls %}
      {%- for tool_call in message.tool_calls %}
          {%- if tool_call.function is defined %}
              {%- set tool_call = tool_call.function %}
          {%- endif %}
          {{-'<tool_call>\n' }}
          {{- '{"name": "'+ tool_call.name + '", "arguments":' }}
          {%- if tool_call.arguments is string -%}
          {{ tool_call.arguments }}
          {%- else -%}
          {{ tool_call.arguments | tojson }}
          {%- endif -%}
          {{- '}' }}
          {{- '\n</tool_call>\n' }}
      {%- endfor %}
    {%- endif %}
    {%- if not loop.last %}
      {{- '<|im_end|>' + '\n' }}
    {%- else %}
      {{- '<|im_end|>' }}
    {%- endif %}
  {%- elif message['role'] == 'tool' %}
    {# Tool responses treated as user messages #}
    {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
        {{- '<|im_start|>user' }}
    {%- endif %}
    {{- '\n<tool_response>\n' + message['content'] + '\n</tool_response>' }}
    {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
  {%- endif %}
  {# --- Add generation prompt after last message if requested --- #}
  {%- if loop.last and add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
  {%- endif %}
{%- endfor %}"""


def resolve_context_path(context_type):
    """Resolve the context file path for a given type ('math' or 'text')."""
    base_path = "/home/golikovp/projects/aip-gpekhime/golikovp/Linguistic_traps/experiments/context_saturation/contexts/"
    if context_type == "math":
        path = os.path.join(base_path, "context_math_1M.json")
    else:
        path = os.path.join(base_path, "context_text_1M.json")
    
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Required context file not found: {path}")


def generate_trimmed_context(tokenizer, context_path, target_size):
    """Binary search to find exact message count for target_size."""
    with open(context_path, 'r') as f:
        messages = json.load(f)
    
    # Strip any existing system prompts from the master file to avoid duplicates
    messages = [m for m in messages if m['role'] != 'system']
    
    # Phase 1: Binary search on full messages
    low = 0
    high = len(messages)
    best_k = 0
    
    system_msg = {"role": "system", "content": BASELINE_SYSTEM_PROMPT}

    print(f"  Phase 1: Binary search on {len(messages)} full messages for {target_size} tokens...")
    
    while low <= high:
        mid = (low + high) // 2
        test_chunk = [system_msg] + messages[:mid]
        
        # Render to string first, then encode (avoids bugs in some tokenize=True implementations)
        rendered = tokenizer.apply_chat_template(test_chunk, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
        count = len(tokens)
        
        if (mid > 0 or len(system_msg['content']) > 0) and count <= 2:
             # Diagnostic render to see what's happening
             rendered = tokenizer.apply_chat_template(test_chunk, tokenize=False, add_generation_prompt=False)
             # If we have content but only get 2 tokens (BOS/EOS), the tokenizer/template is failing.
             raise ValueError(
                 f"Tokenizer returned only {count} tokens for {mid+1} messages. "
                 f"Rendered length: {len(rendered)}. "
                 f"Preview: {rendered[:100]!r}... "
                 f"This usually means the chat template is missing or failing for model '{tokenizer.name_or_path}'."
             )

        if count <= target_size:
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1

    result_msgs = [system_msg] + messages[:best_k]
    rendered_final = tokenizer.apply_chat_template(result_msgs, tokenize=False, add_generation_prompt=False)
    current_tokens = len(tokenizer.encode(rendered_final, add_special_tokens=False))
    
    # Phase 2: Word-level truncation of the NEXT message if we are still under target
    if current_tokens < target_size and best_k < len(messages):
        print(f"  Phase 2: Truncating next message (words) to fill remaining {target_size - current_tokens} tokens...")
        next_msg = messages[best_k]
        words = next_msg['content'].split()
        
        low_w = 0
        high_w = len(words)
        best_w = 0
        
        while low_w <= high_w:
            mid_w = (low_w + high_w) // 2
            partial_content = " ".join(words[:mid_w])
            test_chunk = result_msgs + [{"role": next_msg['role'], "content": partial_content}]
            rendered_test = tokenizer.apply_chat_template(test_chunk, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer.encode(rendered_test, add_special_tokens=False)
            
            if len(tokens) <= target_size:
                best_w = mid_w
                low_w = mid_w + 1
            else:
                high_w = mid_w - 1
        
        if best_w > 0:
            result_msgs.append({"role": next_msg['role'], "content": " ".join(words[:best_w])})

    rendered_absolute = tokenizer.apply_chat_template(result_msgs, tokenize=False, add_generation_prompt=False)
    final_count = len(tokenizer.encode(rendered_absolute, add_special_tokens=False))
    print(f"  Final context built: {len(result_msgs)} messages. Total tokens: {final_count} (Target: {target_size})")
    
    return result_msgs


def run_context_eval(context_type, trimmed_context, dataset, tokenizer, llm, sampling_params, args):
    """
    Prepare all prompt token IDs and metadata upfront.
    Returns (all_inputs, metadata).
    """
    print(f"\n{'='*60}")
    print(f"  Running context evaluation: {context_type.upper()}")
    print(f"{'='*60}")

    # Calculate context tokens once
    common_context_str = tokenizer.apply_chat_template(trimmed_context, tokenize=False, add_generation_prompt=False)
    context_token_count = len(tokenizer.encode(common_context_str, add_special_tokens=False))

    all_inputs = []
    metadata = []

    for i, example in enumerate(dataset):
        cleaned_problem = remove_latex_comments(example['problem'])
        user_prompt, _ = get_prompts(cleaned_problem, 'baseline')
        user_prompt = "Solve the following problem using regular mathematics.\n" + user_prompt

        full_conversation = trimmed_context + [{"role": "user", "content": user_prompt}]

        # Render then encode for final prompt
        final_prompt_str = tokenizer.apply_chat_template(full_conversation, tokenize=False, add_generation_prompt=True)
        final_input_ids = tokenizer.encode(final_prompt_str, add_special_tokens=False)

        for sample_idx in range(args.n_samples):
            all_inputs.append(prompt_ids)
            metadata.append({
                "id": example.get('id', i),
                "sample_idx": sample_idx,
                "post_context_prompt": user_prompt,
                "ground_truth": example['answer'],
            })
    return all_inputs, metadata


def run_context_eval(context_type, all_inputs, metadata, context_token_count, common_context_str, llm, sampling_params, args):
    """
    Run evaluation using pre-prepared token IDs.
    """
    print(f"\n{'='*60}")
    print(f"  Running context evaluation: {context_type.upper()}")
    print(f"{'='*60}")

    # Generate
    print(f"Generating answers for {len(all_inputs)} prompts using prompt_token_ids...")

    if not args.dry:
        # Pass IDs directly to vLLM to avoid redundant processing
        # Note: vLLM generate accepts prompt_token_ids as a list of lists of IDs.
        outputs = llm.generate(prompt_token_ids=all_inputs, sampling_params=sampling_params)
    else:
        print("Dry run: Skipping generation.")
        outputs = []
        class MockOutput:
            def __init__(self, text):
                self.outputs = [type('obj', (object,), {'text': text, 'token_ids': [0]*10})]
        for _ in all_inputs:
            outputs.append(MockOutput("Mock Answer \\boxed{0}"))

    results = []
    stats = {"correct": 0, "total": 0, "failures": 0}

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        output_token_ids = output.outputs[0].token_ids
        output_len = len(output_token_ids)

        meta = metadata[i]

        extracted, is_correct = extract_and_grade(generated_text, meta['ground_truth'])

        results.append({
            "id": meta['id'],
            "sample_idx": meta.get('sample_idx', 0),
            "output": generated_text,
            "post_context_prompt": meta['post_context_prompt'],
            "extracted": extracted,
            "ground_truth": meta['ground_truth'],
            "correct": is_correct,
            "system_prompt": BASELINE_SYSTEM_PROMPT,
            "temperature": 0.7,
            "context_win_total": args.context_win_total,
            "distractor_token_count": context_token_count,
            "max_output_tokens": args.max_output_tokens,
            "model_output_token_count": output_len
        })

        stats["total"] += 1
        if is_correct: stats["correct"] += 1
        else: stats["failures"] += 1

    acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    print(f"\n--- {context_type.upper()} Results ---")
    print(f"Accuracy: {acc:.2%} ({stats['correct']}/{stats['total']})")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace('/', '_')
    filename = f"results_predef_{context_type}_{context_token_count}_{safe_model}_{timestamp}.json"

    safe_dataset = args.dataset.replace('/', '_')
    dirs = f"context_saturation/results_context/{safe_model}/{safe_dataset}"
    os.makedirs(dirs, exist_ok=True)
    out_path = os.path.join(dirs, filename)

    final_output = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "context_math_file": args.context_math_file,
            "context_text_file": args.context_text_file,
            "context_token_count": context_token_count,
            "context_type": context_type,
            "common_context": common_context_str
        },
        "statistics": stats,
        "results": results
    }

    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved {context_type} results to {out_path}")

    return stats, results, out_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate with Predefined Context Saturation (Math & Text)")
    parser.add_argument("--model", type=str, default="tiiuae/Falcon-H1R-7B", help="Model name/path")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/aime_2024", help="Dataset path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--n_samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--context_math_file", type=str, required=True, help="Path to pre-trimmed MATH context JSON")
    parser.add_argument("--context_text_file", type=str, required=True, help="Path to pre-trimmed TEXT context JSON")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--context_win_total", type=int, default=65536, help="Total context window capacity")
    parser.add_argument("--max_output_tokens", type=int, default=4096, help="Max generated tokens")
    parser.add_argument("--dry", action="store_true", help="Dry run")

    args = parser.parse_args()

    # context_types = ['math', 'text'] # Already defined at top of main scope if needed, but let's ensure it's here
    context_types = ['math', 'text']

    # 1. Load Tokenizer FIRST (independently of vLLM to allow context prep before model load)
    print(f"Loading tokenizer for model: {args.model}...")
    from transformers import AutoTokenizer
    # No fallback: throw error if model tokenizer can't be loaded
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Tokenizer loaded. Class: {tokenizer.__class__.__name__}, Vocab size: {tokenizer.vocab_size}")
    
    # Force explicit native chat template for Falcon-H1R models
    # We do this because the Hub config often has it as null, which may trigger 
    # a broken default in some transformers versions.
    if "Falcon-H1R" in args.model:
        print(f"FORCING explicit native chat template for reasoning model '{args.model}'...")
        tokenizer.chat_template = FALCON_CHAT_TEMPLATE
    elif tokenizer.chat_template is None:
        print(f"Warning: No chat template found for model '{args.model}'.")

    # 2. Prepare contexts at the very top (before heavy GPU load)
    print(f"\n{'='*60}")
    print(f"  LOADING CONTEXTS")
    print(f"{'='*60}")
    
    prepped_contexts = {}
    for context_type in context_types:
        try:
            print(f"Preparing {context_type} context...")
            context_path = resolve_context_path(context_type)
            prepped_contexts[context_type] = generate_trimmed_context(tokenizer, context_path, args.context_size)
        except Exception as e:
            print(f"Error preparing {context_type} context: {e}")
            exit(1)

    # 3. Load Dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # 4. Initialize vLLM (only after everything else is ready)
    llm = None
    sampling_params = None

    if not args.dry:
        from vllm import LLM, SamplingParams
        print(f"Initializing vLLM with model: {args.model}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.num_gpus,
            trust_remote_code=True,
            max_model_len=args.context_win_total,
            dtype="bfloat16"
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_output_tokens)
    else:
        print("Dry run: Skipping vLLM initialization.")

    # Run evaluation for each context type
    all_stats = {}
    all_paths = {}

    for context_type in context_types:
        data = prepped_eval_data[context_type]
        stats, results, out_path = run_context_eval(
            context_type=context_type,
            all_inputs=data["inputs"],
            metadata=data["metadata"],
            context_token_count=data["token_count"],
            common_context_str=data["common_str"],
            llm=llm,
            sampling_params=sampling_params,
            args=args
        )
        if stats is not None:
            all_stats[context_type] = stats
            all_paths[context_type] = out_path

    # Print comparison summary
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Type':<8} | {'Accuracy':<18} | {'Correct':<10} | {'Total':<6}")
    print(f"{'-'*8}-+-{'-'*18}-+-{'-'*10}-+-{'-'*6}")

    for ct, st in all_stats.items():
        acc = st["correct"] / st["total"] if st["total"] > 0 else 0
        print(f"{ct:<8} | {acc:<18.2%} | {st['correct']:<10} | {st['total']:<6}")

    print()
    for ct, path in all_paths.items():
        print(f"  {ct}: {path}")

    # Save combined comparison
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace('/', '_')
    safe_dataset = args.dataset.replace('/', '_')
    dirs = f"context_saturation/results_context/{safe_model}/{safe_dataset}"
    os.makedirs(dirs, exist_ok=True)
    comparison_path = os.path.join(dirs, f"comparison_{safe_model}_{timestamp}.json")

    comparison_output = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "context_math_file": args.context_math_file,
            "context_text_file": args.context_text_file,
            "context_types": list(all_stats.keys()),
        },
        "comparison": {ct: {"statistics": st, "results_file": all_paths[ct]} for ct, st in all_stats.items()},
    }

    with open(comparison_path, 'w') as f:
        json.dump(comparison_output, f, indent=2)
    print(f"\nSaved comparison to {comparison_path}")

if __name__ == "__main__":
    main()
