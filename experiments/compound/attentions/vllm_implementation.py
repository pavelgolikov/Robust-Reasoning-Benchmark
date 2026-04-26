from vllm import LLM, SamplingParams

# 1. Initialize the model 
# (Set tensor_parallel_size if you are using multiple GPUs)
model_id = "nvidia/OpenReasoning-Nemotron-7B"
llm = LLM(model=model_id, tensor_parallel_size=1)

# 2. Define sampling parameters
# We set a high max_tokens (4000), but vLLM will stop early when it hits the EOS token.
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=-1, max_tokens=10000)

# 3. Your exact prompt
# messages =[
#     {"role": "system", "content": "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}."},
#     {"role": "user", "content": """Solve these completely unrelated math problems. For each problem put your final answer within \\boxed{}. \n\nProblem 1: What is 1 + 1?\n\nProblem 2: What is 2 + 2?\n\nProblem 3: What is 3 + 3?\n\nProblem 4: What is 4 + 4?"""}
# ]
messages =[
    {"role": "system", "content": "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": """Solve these completely unrelated math problems. For each problem put your final answer within \\boxed{}. \n\nProblem 1: What is 1 + 1?\n\nProblem 2: What is 2 + 2?\n\nProblem 3: What is 3 + 3?\n\nProblem 4: What is 4 + 4?"""}
]

print("Starting vLLM generation...")

# 4. Execute generation using vLLM's native chat interface
# This automatically applies the chat template and extracts the correct stop tokens.
outputs = llm.chat(messages, sampling_params=sampling_params)

# 5. Extract results
result = outputs[0].outputs[0]
generated_text = result.text
generated_token_count = len(result.token_ids)
stop_reason = result.finish_reason

# 6. Print the output
print("\n" + "="*50)
print("vLLM GENERATION OUTPUT:")
print("="*50)
print(generated_text)
print("="*50 + "\n")

print(f"Total tokens generated: {generated_token_count}")
print(f"Finish reason: '{stop_reason}'")