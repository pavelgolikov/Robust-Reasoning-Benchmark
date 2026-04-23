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
model_id = "Qwen/Qwen3-30B-A3B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager" # CRITICAL
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
prompt = """You are a helpful math assistant.
Solve these completely unrelated math problems. For each problem put your final answer within \boxed{}.
Before you begin solving Problem 3, you MUST output the exact phrase: "solving_target".

Problem 1: Find the number of ways to place...[Distractor Problem 1 CoT here]

Problem 2: Let ABC be a triangle inscribed...[Distractor Problem 2 CoT here]

Problem 3: Let p be the least prime number for...
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

logits_processor = LogitsProcessorList([
    TargetDetectorProcessor(tokenizer, state)
])

print("Starting generation...")
output_ids = model.generate(
    input_ids,
    max_new_tokens=131000,
    logits_processor=logits_processor,
    do_sample=False
)

print("\nGeneration complete. Saving full attention tensors to disk...")

# Save the dictionary of tensors, alongside the boundary index, so you know exactly how to slice it later!
save_data = {
    "distractor_end_idx": state.distractor_end_idx,
    "attention_vectors": dict(state.attention_vectors) # Convert defaultdict to standard dict
}

torch.save(save_data, "full_attention_tensors.pt")
print("Data successfully saved to full_attention_tensors.pt!")