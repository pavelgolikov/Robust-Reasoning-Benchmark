import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from collections import defaultdict

# ==========================================
# 1. EXPERIMENT STATE TRACKER
# ==========================================
class ExperimentState:
    def __init__(self):
        self.in_target_phase = False
        self.distractor_end_idx = 0
        
        # Dictionary to hold lists of tensors: layer_idx -> [tensor_step_1, tensor_step_2, ...]
        self.attention_vectors = defaultdict(list)

state = ExperimentState()

# ==========================================
# 2. THE DETECTOR (Logits Processor)
# ==========================================
class TargetDetectorProcessor(LogitsProcessor):
    def __init__(self, tokenizer, state_tracker):
        self.tokenizer = tokenizer
        self.state = state_tracker
        self.flag_phrase = "solving_target"

    def __call__(self, input_ids, scores):
        if not self.state.in_target_phase:
            recent_text = self.tokenizer.decode(input_ids[0, -10:]).lower()
            if self.flag_phrase in recent_text:
                self.state.in_target_phase = True
                self.state.distractor_end_idx = input_ids.shape[1]
                print(f"\n[Detector] Target Phase Initiated! Distractor boundary at token {self.state.distractor_end_idx}")
        return scores

# ==========================================
# 3. INITIALIZATION
# ==========================================
model_id = "nvidia/OpenReasoning-Nemotron-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager" # Flash attention does not compute attention scores directly, so we use eager mode
)

# ==========================================
# 4. THE ATTENTION HOOK (Monkey-Patching)
# ==========================================
AttentionClass = type(model.model.layers[0].self_attn)
original_forward = AttentionClass.forward

def oom_safe_attention_hook(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    
    outputs = original_forward(
        self, 
        hidden_states=hidden_states, 
        attention_mask=attention_mask, 
        position_ids=position_ids, 
        past_key_value=past_key_value, 
        output_attentions=True, 
        use_cache=use_cache, 
        **kwargs
    )
    
    attn_output = outputs[0]
    attn_weights = outputs[1] # Shape: (batch, heads, q_len, total_seq_len)
    
    q_len = attn_weights.shape[2]
    print(f"Attention weights with shape: {attn_weights.shape}")
    
    # Only capture during decoding (q_len == 1) AND when we are in the target phase
    if q_len == 1 and state.in_target_phase:
        # attn_weights shape is (1, num_heads, 1, current_context_length)
        # Squeeze out the batch and q_len dimensions -> resulting shape: (num_heads, current_context_length)
        vector_to_save = attn_weights[0, :, 0, :]
        
        # CRITICAL PRECAUTIONS FOR MEMORY:
        # 1. .detach() removes it from the autograd graph
        # 2. .cpu() moves it from VRAM to standard RAM
        # 3. .to(torch.float16) compresses it to save space
        vector_to_save = vector_to_save.detach().cpu().to(torch.float16)
        print(f"vector_to_save shape: {vector_to_save.shape}, dtype: {vector_to_save.dtype}")
        
        state.attention_vectors[self.layer_idx].append(vector_to_save)
    
    # IMMEDIATE DELETION TO PREVENT VRAM OOM
    del attn_weights
    
    if len(outputs) == 3:
        return (attn_output, None, outputs[2])
    return (attn_output, None)

AttentionClass.forward = oom_safe_attention_hook

# ==========================================
# 5. EXECUTION & SAVING
# ==========================================
messages =[
    {"role": "system", "content": "You are a helpful math assistant. Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": """Solve these completely unrelated math problems. For each problem put your final answer within \\boxed{}. \n\nProblem 1: What is 1 + 1?\n\nProblem 2: What is 2 + 2?\n\nProblem 3: What is 3 + 3?\n\nProblem 4: What is 4 + 4?"""}
]

# {"role": "user", "content": """Solve these completely unrelated math problems. For each problem put your final answer within \\boxed{}. Before you begin solving last problem, you MUST output the exact phrase: solving_target.\n\nProblem 1:\nAlice and Bob play the following game. A stack of $n$ tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either $1$ token or $4$ tokens from the stack. Whoever removes the last token wins. Find the number of positive integers $n$ less than or equal to $2024$ for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play.\n\nProblem 2:\nLet ABCDEF be a convex equilateral hexagon in which all pairs of opposite sides are parallel. The triangle whose sides are extensions of segments AB, CD, and EF has side lengths 200, 240, and 300. Find the side length of the hexagon."""}

# 1. Get the raw formatted string using the model's chat template
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False, # Don't turn it into tensors yet
    add_generation_prompt=True
)

# 2. Tokenize the string properly, which gives us a dict with input_ids and attention_mask
inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

logits_processor = LogitsProcessorList([
    TargetDetectorProcessor(tokenizer, state)
])

print("Starting generation...")

# 3. Use **inputs to unpack the dictionary correctly!
output_ids = model.generate(
    **inputs, 
    max_new_tokens=10000,
    logits_processor=logits_processor,
    # eos_token_id=tokenizer.eos_token_id,          # <-- THIS is the critical native fix
    eos_token_id=[151643, 151645],
    pad_token_id=tokenizer.eos_token_id,  # <-- Prevents Hugging Face warning logs
    do_sample=True,
    temperature=0.6,
    top_k=0,
    top_p=0.95
)

# print out the generated text to verify everything is working as expected
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print("\nGenerated Text:\n", generated_text)
print("Token cound of generated text (including prompt):", len(output_ids[0]))

print("\nGeneration complete. Saving full attention tensors to disk...")

# Save the dictionary of tensors, alongside the boundary index, so you know exactly how to slice it later!
save_data = {
    "distractor_end_idx": state.distractor_end_idx,
    "attention_vectors": dict(state.attention_vectors) # Convert defaultdict to standard dict
}

torch.save(save_data, "full_attention_tensors.pt")
print("Data successfully saved to full_attention_tensors.pt!")