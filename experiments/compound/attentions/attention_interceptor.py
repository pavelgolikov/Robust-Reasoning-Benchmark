import torch
import torch.nn.functional as F
import math

# Global dictionary to store the extracted [Heads, Target_Tokens] dilution vectors
# Keys will be layer index (int), Values will be CPU tensors
dilution_results = {}

def get_attention_interceptor(layer_idx: int, target_start_idx: int, chunk_size: int = 500, model_type: str = "llama"):
    """
    Creates a custom forward function that replaces the native attention forward.
    This intercepts Q and K post-RoPE, computes the target attention blocks
    in VRAM-safe chunks, extracts the dilution scores to CPU, and resumes normally.
    """
    if model_type == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    else:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        bsz, q_len, _ = hidden_states.size()

        # 1. Standard Q, K, V Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 2. Apply Rotary Position Embeddings (RoPE)
        kv_seq_len = key_states.shape[-2]
        
        # Position IDs are strictly required for accurate RoPE. Fallback to basic arange if not passed.
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).view(-1, kv_seq_len)
            
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat K/V for Grouped Query Attention (GQA) if applicable
        if hasattr(self, "num_key_value_groups") and self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # 3. ON-THE-FLY DILUTION COMPUTATION (Memory Safe)
        # We only compute this if our forward pass includes the target indices
        if q_len > target_start_idx:
            
            # Isolate queries belonging ONLY to the Target Problem
            Q_target = query_states[:, :, target_start_idx:, :]
            num_target_tokens = Q_target.shape[2]
            
            # Pre-allocate CPU tensor to hold results (saves VRAM)
            layer_scores = torch.zeros((bsz, self.num_heads, num_target_tokens), device='cpu')

            # Process in chunks to prevent O(N^2) OOM crashes
            for chunk_start in range(0, num_target_tokens, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_target_tokens)
                Q_chunk = Q_target[:, :, chunk_start:chunk_end, :]
                
                # Compute raw unnormalized attention scores (Q * K^T) for this chunk
                attn_weights = torch.matmul(Q_chunk, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                
                # Create and apply Causal Mask mathematically
                global_q_indices = torch.arange(
                    target_start_idx + chunk_start, 
                    target_start_idx + chunk_end, 
                    device=attn_weights.device
                ).view(-1, 1)
                global_k_indices = torch.arange(kv_seq_len, device=attn_weights.device).view(1, -1)
                
                causal_mask = (global_k_indices > global_q_indices).unsqueeze(0).unsqueeze(0)
                attn_weights.masked_fill_(causal_mask, float('-inf'))
                
                # Apply Softmax to get exact probability distributions
                attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                
                # Aggregate: Sum probability mass looking specifically at Distractor Indices (0 to target_start_idx)
                distractor_mass = attn_probs[:, :, :, :target_start_idx].sum(dim=-1)
                
                # Move directly to CPU
                layer_scores[:, :, chunk_start:chunk_end] = distractor_mass.detach().cpu()
                
                # Immediately destroy massive intermediate tensors to keep VRAM flat
                del attn_weights, causal_mask, attn_probs, distractor_mass
                torch.cuda.empty_cache()

            # Save the metric to the global dictionary (squeeze batch dim)
            dilution_results[layer_idx] = layer_scores.squeeze(0)

        # 4. RESUME NATIVE FORWARD PASS
        # Hand computation back to highly-optimized PyTorch fused kernel for standard output
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=True
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return custom_forward

def attach_dilution_interceptors(model, target_start_idx: int, chunk_size: int = 500, model_type: str = "llama"):
    """
    Applies the custom forward hook (interceptor) to every layer in the model.
    """
    # Clear any previous results
    global dilution_results
    dilution_results.clear()
    
    for i, layer in enumerate(model.model.layers):
        # Override the self_attn.forward method by binding the custom function to the object
        layer.self_attn.forward = get_attention_interceptor(
            layer_idx=i, 
            target_start_idx=target_start_idx, 
            chunk_size=chunk_size,
            model_type=model_type
        ).__get__(layer.self_attn, type(layer.self_attn))