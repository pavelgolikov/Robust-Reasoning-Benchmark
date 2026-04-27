import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# Global dictionary to store the extracted [Heads, Target_Tokens] dilution vectors
# Keys will be layer index (int), Values will be CPU tensors
dilution_results = {}

def get_attention_interceptor(system_end_idx: int, target_start_idx: int, chunk_size: int = 500, model_type: str = "qwen3"):
    """
    Creates a custom forward function that replaces the native Attention.forward.
    Handles both Qwen3 (Q/K RMSNorms) and Qwen2 architectures cleanly via model_type routing.
    """
    if "qwen2" in model_type.lower():
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward
    else:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward

    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # 1. Projections AND Normalizations (Conditional on architecture)
        if "qwen3" in model_type.lower():
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 2. Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[2]

        # 3. ON-THE-FLY DILUTION COMPUTATION (Memory Safe)
        if q_len > target_start_idx:
            # Explicitly repeat KV heads for GQA so manual torch.matmul works correctly
            key_states_expanded = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            
            Q_target = query_states[:, :, target_start_idx:, :]
            num_target_tokens = Q_target.shape[2]
            
            layer_scores = {
                "system": torch.zeros((bsz, num_heads, num_target_tokens), device='cpu'),
                "distractor": torch.zeros((bsz, num_heads, num_target_tokens), device='cpu'),
                "target": torch.zeros((bsz, num_heads, num_target_tokens), device='cpu')
            }

            for chunk_start in range(0, num_target_tokens, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_target_tokens)
                Q_chunk = Q_target[:, :, chunk_start:chunk_end, :]
                
                # Compute raw unnormalized attention scores
                attn_weights = torch.matmul(Q_chunk, key_states_expanded.transpose(2, 3)) * self.scaling
                
                # Create and apply Causal Mask
                global_q_indices = torch.arange(
                    target_start_idx + chunk_start, 
                    target_start_idx + chunk_end, 
                    device=attn_weights.device
                ).view(-1, 1)
                global_k_indices = torch.arange(kv_seq_len, device=attn_weights.device).view(1, -1)
                
                causal_mask = (global_k_indices > global_q_indices).unsqueeze(0).unsqueeze(0)
                attn_weights.masked_fill_(causal_mask, float('-inf'))
                
                # Layer-specific Sliding Window Masking
                if getattr(self, "sliding_window", None) is not None:
                    sliding_window_mask = (global_q_indices - global_k_indices > self.sliding_window).unsqueeze(0).unsqueeze(0)
                    attn_weights.masked_fill_(sliding_window_mask, float('-inf'))

                # Apply Softmax
                attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                
                # Aggregate probability mass
                system_mass = attn_probs[:, :, :, :system_end_idx].sum(dim=-1)
                distractor_mass = attn_probs[:, :, :, system_end_idx:target_start_idx].sum(dim=-1)
                target_mass = attn_probs[:, :, :, target_start_idx:].sum(dim=-1)
                
                layer_scores["system"][:, :, chunk_start:chunk_end] = system_mass.detach().cpu()
                layer_scores["distractor"][:, :, chunk_start:chunk_end] = distractor_mass.detach().cpu()
                layer_scores["target"][:, :, chunk_start:chunk_end] = target_mass.detach().cpu()
                
                # Free VRAM
                del attn_weights, causal_mask, attn_probs, system_mass, distractor_mass, target_mass
                if getattr(self, "sliding_window", None) is not None:
                    del sliding_window_mask
                torch.cuda.empty_cache()

            dilution_results[self.layer_idx] = {k: v.squeeze(0) for k, v in layer_scores.items()}

        # 4. RESUME NATIVE FORWARD PASS
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # CRITICAL FIX for the RuntimeError: 
        # This properly reshapes (bsz, q_len, num_heads, head_dim) into (bsz, q_len, hidden_size)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights

    return custom_forward

def attach_dilution_interceptors(model, system_end_idx: int, target_start_idx: int, chunk_size: int = 500, model_type: str = "qwen3"):
    """
    Applies the forward hook interceptor to every layer in the model.
    """
    global dilution_results
    dilution_results.clear()
    
    for layer in model.model.layers:
        layer.self_attn.forward = get_attention_interceptor(
            system_end_idx=system_end_idx,
            target_start_idx=target_start_idx, 
            chunk_size=chunk_size,
            model_type=model_type
        ).__get__(layer.self_attn, type(layer.self_attn))