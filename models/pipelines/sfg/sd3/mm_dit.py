"""
SFG (Segmentation-free Guidance) implementation for Stable Diffusion 3.

Reference: "Segmentation-free guidance for text-to-image diffusion models" (CVPR 2024 Workshop)
Authors: Kambiz Azarian, Debasmit Das, Qiqi Hou, Fatih Porikli

Key mechanism:
- For each image patch, find the text token with highest cross-attention weight
- Multiply that token's attention weight by -a (a=10 by default)
- This "excludes" the most relevant concept from the negative prompt for each patch

Key difference from CFG:
- CFG: ε(z,c) - ε(z,∅)  (conditional vs unconditional)
- SFG: ε(z,c) - ε̄(z,c)  (conditional vs attention-modified conditional)
  Both branches use conditional input! Only attention weights differ.
"""

from typing import Optional
import torch
import math
import torch.nn.functional as F


class SFGJointAttnProcessor:
    """
    Attention processor for SFG (Segmentation-free Guidance) adapted for SD3's JointAttention.
    
    This processor modifies image-to-text cross-attention weights for the perturbed branch:
    - For each image patch, find the text token with argmax attention (excluding BOS)
    - Multiply that token's attention weight by -a (sfg_scale)
    
    Args:
        sfg_scale (float): Scale factor for attention modification (a in paper, default 10.0).
            The max attention weight is multiplied by -sfg_scale.
        exclude_bos (bool): Whether to exclude BOS token from argmax selection.
            Paper recommends True because BOS has high attention but negligible value.
    """
    
    def __init__(self, sfg_scale: float = 10.0, exclude_bos: bool = True):
        self.sfg_scale = sfg_scale
        self.exclude_bos = exclude_bos

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Apply SFG-enabled joint attention.
        
        In SFG mode, batch contains [cond_original, cond_perturbed].
        Both use the same conditional input, but the perturbed branch has modified attention.
        
        Process:
        1. Compute Q, K, V for image and text
        2. For perturbed branch: manually compute attention, modify weights, then apply
        3. For original branch: use standard attention
        4. Concatenate and return
        """
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        
        # Step 1: Compute image projections (Q, K, V from hidden_states)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # Reshape to multi-head format: (batch, seq_len, dim) -> (batch, heads, seq_len, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Apply query/key normalization (SD3 uses RMSNorm)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        img_seq_len = hidden_states.shape[1]
        
        # Step 2: Process text/context Q, K, V
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            # Concatenate image and text Q, K, V for joint attention
            query_joint = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key_joint = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value_joint = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
        else:
            query_joint = query
            key_joint = key
            value_joint = value
            text_seq_len = 0
        
        # Step 3: SFG - Apply attention modification to perturbed branch (second half of batch)
        # Batch structure: [original, perturbed]
        half_batch = batch_size // 2
        
        # Split into original and perturbed branches
        query_org, query_ptb = query_joint[:half_batch], query_joint[half_batch:]
        key_org, key_ptb = key_joint[:half_batch], key_joint[half_batch:]
        value_org, value_ptb = value_joint[:half_batch], value_joint[half_batch:]
        
        # Original branch: standard attention
        hidden_states_org = F.scaled_dot_product_attention(
            query_org, key_org, value_org, dropout_p=0.0, is_causal=False
        )
        
        # Perturbed branch: manually compute attention with modification
        # We need to modify the image-to-text attention sub-matrix
        scale = 1.0 / math.sqrt(head_dim)
        
        # Compute full attention scores: (batch, heads, seq, seq)
        # seq = img_seq + text_seq
        attn_scores_ptb = torch.matmul(query_ptb, key_ptb.transpose(-2, -1)) * scale
        
        if encoder_hidden_states is not None:
            # Extract image-to-text attention sub-matrix
            # Shape: (half_batch, heads, img_seq_len, text_seq_len)
            img_to_text_attn = attn_scores_ptb[:, :, :img_seq_len, img_seq_len:]
            
            # For each image patch, find the text token with max attention (excluding BOS)
            # Average across heads for finding argmax
            img_to_text_attn_avg = img_to_text_attn.mean(dim=1)  # (half_batch, img_seq, text_seq)
            
            if self.exclude_bos and text_seq_len > 1:
                # Exclude first token (BOS) from argmax
                # Set BOS attention to -inf for argmax selection only
                img_to_text_attn_for_argmax = img_to_text_attn_avg.clone()
                img_to_text_attn_for_argmax[:, :, 0] = float('-inf')
                max_text_indices = img_to_text_attn_for_argmax.argmax(dim=-1)  # (half_batch, img_seq)
            else:
                max_text_indices = img_to_text_attn_avg.argmax(dim=-1)  # (half_batch, img_seq)
            
            # Create modification mask
            # For each image patch, multiply the max text token's attention by -sfg_scale
            # Shape: (half_batch, heads, img_seq_len, text_seq_len)
            modification_mask = torch.zeros_like(img_to_text_attn)
            
            # Create indices for scatter
            batch_indices = torch.arange(half_batch, device=hidden_states.device).view(-1, 1).expand(-1, img_seq_len)
            img_indices = torch.arange(img_seq_len, device=hidden_states.device).view(1, -1).expand(half_batch, -1)
            
            # Apply modification: multiply by -a means new_weight = old_weight * (-a)
            # So the modification is: old_weight * (-a) - old_weight = old_weight * (-a - 1)
            # But actually, we want: new_weight = old_weight * (-a)
            # So we add: old_weight * (-a - 1) to the original
            
            # Get the original attention values at max positions
            # Shape: (half_batch, img_seq_len)
            original_attn_values = img_to_text_attn_avg[batch_indices, img_indices, max_text_indices]
            
            # The modification: we want attn[i,j,max_k] *= -a
            # Which means attn[i,j,max_k] = attn[i,j,max_k] * (-a)
            # Additive change: attn[i,j,max_k] * (-a - 1)
            
            # Apply to all heads
            for h in range(attn.heads):
                # Get attention values for this head at max positions
                head_attn = attn_scores_ptb[:half_batch, h, :img_seq_len, img_seq_len:]  # (half_batch, img_seq, text_seq)
                
                # Get original values at max positions for this head
                original_head_values = head_attn[batch_indices, img_indices, max_text_indices]
                
                # Compute additive modification
                additive_mod = original_head_values * (-self.sfg_scale - 1)
                
                # Apply modification using scatter_add
                attn_scores_ptb[:half_batch, h, :img_seq_len, img_seq_len:].scatter_add_(
                    dim=-1,
                    index=max_text_indices.unsqueeze(-1),
                    src=additive_mod.unsqueeze(-1)
                )
        
        # Apply softmax and compute attention output for perturbed branch
        attn_weights_ptb = F.softmax(attn_scores_ptb, dim=-1)
        hidden_states_ptb = torch.matmul(attn_weights_ptb, value_ptb)
        
        # Concatenate original and perturbed outputs
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb], dim=0)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # Step 4: Split outputs back into image and text
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states_out = (
                hidden_states[:, :img_seq_len],
                hidden_states[:, img_seq_len:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
        
        # Step 5: Final output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states_out
        else:
            return hidden_states
