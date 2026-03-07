# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F


class PAGFluxAttnProcessor:
    """
    Attention processor for Flux model with Perturbed Attention Guidance (PAG) without CFG.
    
    This processor implements PAG by creating an identity mask that blocks self-attention 
    between image patches, forcing the model to rely more on text conditioning.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "PAGFluxAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Apply PAG-enabled attention for Flux model.
        
        Args:
            attn: The attention module
            hidden_states: Image latent features
            encoder_hidden_states: Text conditioning features (optional for single blocks)
            attention_mask: Attention mask (optional)
            image_rotary_emb: Rotary position embeddings
            
        Returns:
            Processed hidden states (and encoder hidden states if provided)
        """
        # Determine if this is a joint attention (has encoder_hidden_states)
        is_joint_attn = encoder_hidden_states is not None
        
        # Store identity block size for masking
        # For Flux, when we have encoder_hidden_states, the sequence is concatenated [text, image]
        identity_block_size = hidden_states.shape[1]
        
        # Chunk into original and perturbed paths
        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)
        if is_joint_attn:
            encoder_hidden_states_org, encoder_hidden_states_ptb = encoder_hidden_states.chunk(2)
        
        ################## Original path ##################
        batch_size = hidden_states_org.shape[0]
        
        # Image projections
        query_org = attn.to_q(hidden_states_org)
        key_org = attn.to_k(hidden_states_org)
        value_org = attn.to_v(hidden_states_org)
        
        inner_dim = key_org.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # Reshape to multi-head format
        query_org = query_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_org = key_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_org = value_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Apply normalization
        if attn.norm_q is not None:
            query_org = attn.norm_q(query_org)
        if attn.norm_k is not None:
            key_org = attn.norm_k(key_org)
        
        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query_org = apply_rotary_emb(query_org, image_rotary_emb)
            key_org = apply_rotary_emb(key_org, image_rotary_emb)
        
        # Joint attention: concatenate text features
        if is_joint_attn:
            # Text projections
            encoder_query_org = attn.add_q_proj(encoder_hidden_states_org)
            encoder_key_org = attn.add_k_proj(encoder_hidden_states_org)
            encoder_value_org = attn.add_v_proj(encoder_hidden_states_org)
            
            # Reshape text projections
            encoder_query_org = encoder_query_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key_org = encoder_key_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value_org = encoder_value_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # Apply normalization for text
            if attn.norm_added_q is not None:
                encoder_query_org = attn.norm_added_q(encoder_query_org)
            if attn.norm_added_k is not None:
                encoder_key_org = attn.norm_added_k(encoder_key_org)
            
            # Concatenate image and text
            query_org = torch.cat([encoder_query_org, query_org], dim=2)
            key_org = torch.cat([encoder_key_org, key_org], dim=2)
            value_org = torch.cat([encoder_value_org, value_org], dim=2)
        
        # Compute attention
        hidden_states_org = F.scaled_dot_product_attention(
            query_org, key_org, value_org, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query_org.dtype)
        
        # Split back if joint attention
        if is_joint_attn:
            encoder_hidden_states_org, hidden_states_org = (
                hidden_states_org[:, : encoder_hidden_states.shape[1]],
                hidden_states_org[:, encoder_hidden_states.shape[1] :],
            )
            # Project encoder output
            if not attn.context_pre_only:
                encoder_hidden_states_org = attn.to_add_out(encoder_hidden_states_org)
        
        # Output projection for image features
        if not attn.pre_only:
            hidden_states_org = attn.to_out[0](hidden_states_org)
            hidden_states_org = attn.to_out[1](hidden_states_org)
        
        ################## Perturbed path (with identity mask) ##################
        batch_size = hidden_states_ptb.shape[0]
        
        # Image projections
        query_ptb = attn.to_q(hidden_states_ptb)
        key_ptb = attn.to_k(hidden_states_ptb)
        value_ptb = attn.to_v(hidden_states_ptb)
        
        # Reshape to multi-head format
        query_ptb = query_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_ptb = key_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_ptb = value_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Apply normalization
        if attn.norm_q is not None:
            query_ptb = attn.norm_q(query_ptb)
        if attn.norm_k is not None:
            key_ptb = attn.norm_k(key_ptb)
        
        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query_ptb = apply_rotary_emb(query_ptb, image_rotary_emb)
            key_ptb = apply_rotary_emb(key_ptb, image_rotary_emb)
        
        # Joint attention: concatenate text features
        if is_joint_attn:
            # Text projections
            encoder_query_ptb = attn.add_q_proj(encoder_hidden_states_ptb)
            encoder_key_ptb = attn.add_k_proj(encoder_hidden_states_ptb)
            encoder_value_ptb = attn.add_v_proj(encoder_hidden_states_ptb)
            
            # Reshape text projections
            encoder_query_ptb = encoder_query_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key_ptb = encoder_key_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value_ptb = encoder_value_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # Apply normalization for text
            if attn.norm_added_q is not None:
                encoder_query_ptb = attn.norm_added_q(encoder_query_ptb)
            if attn.norm_added_k is not None:
                encoder_key_ptb = attn.norm_added_k(encoder_key_ptb)
            
            # Concatenate image and text
            query_ptb = torch.cat([encoder_query_ptb, query_ptb], dim=2)
            key_ptb = torch.cat([encoder_key_ptb, key_ptb], dim=2)
            value_ptb = torch.cat([encoder_value_ptb, value_ptb], dim=2)
        
        # Create identity mask for PAG
        # This mask prevents image patches from attending to each other (except diagonal)
        # but allows image-text cross-attention
        seq_len = query_ptb.size(2)
        full_mask = torch.zeros((seq_len, seq_len), device=query_ptb.device, dtype=query_ptb.dtype)
        
        if is_joint_attn:
            # For joint attention, mask only image-to-image attention (lower right block)
            text_seq_len = encoder_hidden_states.shape[1]
            img_start = text_seq_len
            full_mask[img_start:, img_start:] = float("-inf")
            full_mask[img_start:, img_start:].fill_diagonal_(0)
        else:
            # For single blocks (no text), mask all image-to-image attention except diagonal
            full_mask[:, :] = float("-inf")
            full_mask.fill_diagonal_(0)
        
        # Expand mask for batch and heads
        full_mask = full_mask.unsqueeze(0).unsqueeze(0)
        
        # Compute attention with mask
        hidden_states_ptb = F.scaled_dot_product_attention(
            query_ptb, key_ptb, value_ptb, attn_mask=full_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_ptb = hidden_states_ptb.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_ptb = hidden_states_ptb.to(query_ptb.dtype)
        
        # Split back if joint attention
        if is_joint_attn:
            encoder_hidden_states_ptb, hidden_states_ptb = (
                hidden_states_ptb[:, : encoder_hidden_states.shape[1]],
                hidden_states_ptb[:, encoder_hidden_states.shape[1] :],
            )
            # Project encoder output
            if not attn.context_pre_only:
                encoder_hidden_states_ptb = attn.to_add_out(encoder_hidden_states_ptb)
        
        # Output projection for image features
        if not attn.pre_only:
            hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
            hidden_states_ptb = attn.to_out[1](hidden_states_ptb)
        
        ################ Concatenate original and perturbed ###############
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])
        
        if is_joint_attn:
            encoder_hidden_states = torch.cat([encoder_hidden_states_org, encoder_hidden_states_ptb])
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class PAGCFGFluxAttnProcessor:
    """
    Attention processor for Flux model with PAG and CFG enabled.
    
    Handles three branches: unconditional, conditional (original), and conditional (perturbed).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "PAGCFGFluxAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Apply PAG+CFG enabled attention for Flux model.
        
        Processes three branches: [uncond, cond_original, cond_perturbed]
        """
        is_joint_attn = encoder_hidden_states is not None
        identity_block_size = hidden_states.shape[1]
        
        # Chunk into three branches: uncond, cond_org, cond_ptb
        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_org])
        
        if is_joint_attn:
            encoder_hidden_states_uncond, encoder_hidden_states_org, encoder_hidden_states_ptb = encoder_hidden_states.chunk(3)
            encoder_hidden_states_org = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_org])
        
        ################## Original path (uncond + cond) ##################
        batch_size = hidden_states_org.shape[0]
        
        # Image projections
        query_org = attn.to_q(hidden_states_org)
        key_org = attn.to_k(hidden_states_org)
        value_org = attn.to_v(hidden_states_org)
        
        inner_dim = key_org.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # Reshape to multi-head format
        query_org = query_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_org = key_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_org = value_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Apply normalization
        if attn.norm_q is not None:
            query_org = attn.norm_q(query_org)
        if attn.norm_k is not None:
            key_org = attn.norm_k(key_org)
        
        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query_org = apply_rotary_emb(query_org, image_rotary_emb)
            key_org = apply_rotary_emb(key_org, image_rotary_emb)
        
        # Joint attention
        if is_joint_attn:
            encoder_query_org = attn.add_q_proj(encoder_hidden_states_org)
            encoder_key_org = attn.add_k_proj(encoder_hidden_states_org)
            encoder_value_org = attn.add_v_proj(encoder_hidden_states_org)
            
            encoder_query_org = encoder_query_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key_org = encoder_key_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value_org = encoder_value_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                encoder_query_org = attn.norm_added_q(encoder_query_org)
            if attn.norm_added_k is not None:
                encoder_key_org = attn.norm_added_k(encoder_key_org)
            
            query_org = torch.cat([encoder_query_org, query_org], dim=2)
            key_org = torch.cat([encoder_key_org, key_org], dim=2)
            value_org = torch.cat([encoder_value_org, value_org], dim=2)
        
        # Compute attention
        hidden_states_org = F.scaled_dot_product_attention(
            query_org, key_org, value_org, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query_org.dtype)
        
        # Split back if joint attention
        if is_joint_attn:
            encoder_hidden_states_org, hidden_states_org = (
                hidden_states_org[:, : encoder_hidden_states.shape[1]],
                hidden_states_org[:, encoder_hidden_states.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states_org = attn.to_add_out(encoder_hidden_states_org)
        
        # Output projection
        if not attn.pre_only:
            hidden_states_org = attn.to_out[0](hidden_states_org)
            hidden_states_org = attn.to_out[1](hidden_states_org)
        
        ################## Perturbed path ##################
        batch_size = hidden_states_ptb.shape[0]
        
        query_ptb = attn.to_q(hidden_states_ptb)
        key_ptb = attn.to_k(hidden_states_ptb)
        value_ptb = attn.to_v(hidden_states_ptb)
        
        query_ptb = query_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_ptb = key_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_ptb = value_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if attn.norm_q is not None:
            query_ptb = attn.norm_q(query_ptb)
        if attn.norm_k is not None:
            key_ptb = attn.norm_k(key_ptb)
        
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query_ptb = apply_rotary_emb(query_ptb, image_rotary_emb)
            key_ptb = apply_rotary_emb(key_ptb, image_rotary_emb)
        
        if is_joint_attn:
            encoder_query_ptb = attn.add_q_proj(encoder_hidden_states_ptb)
            encoder_key_ptb = attn.add_k_proj(encoder_hidden_states_ptb)
            encoder_value_ptb = attn.add_v_proj(encoder_hidden_states_ptb)
            
            encoder_query_ptb = encoder_query_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_key_ptb = encoder_key_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_value_ptb = encoder_value_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                encoder_query_ptb = attn.norm_added_q(encoder_query_ptb)
            if attn.norm_added_k is not None:
                encoder_key_ptb = attn.norm_added_k(encoder_key_ptb)
            
            query_ptb = torch.cat([encoder_query_ptb, query_ptb], dim=2)
            key_ptb = torch.cat([encoder_key_ptb, key_ptb], dim=2)
            value_ptb = torch.cat([encoder_value_ptb, value_ptb], dim=2)
        
        # Create identity mask
        seq_len = query_ptb.size(2)
        full_mask = torch.zeros((seq_len, seq_len), device=query_ptb.device, dtype=query_ptb.dtype)
        
        if is_joint_attn:
            text_seq_len = encoder_hidden_states.shape[1]
            img_start = text_seq_len
            full_mask[img_start:, img_start:] = float("-inf")
            full_mask[img_start:, img_start:].fill_diagonal_(0)
        else:
            full_mask[:, :] = float("-inf")
            full_mask.fill_diagonal_(0)
        
        full_mask = full_mask.unsqueeze(0).unsqueeze(0)
        
        # Compute attention with mask
        hidden_states_ptb = F.scaled_dot_product_attention(
            query_ptb, key_ptb, value_ptb, attn_mask=full_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_ptb = hidden_states_ptb.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_ptb = hidden_states_ptb.to(query_ptb.dtype)
        
        if is_joint_attn:
            encoder_hidden_states_ptb, hidden_states_ptb = (
                hidden_states_ptb[:, : encoder_hidden_states.shape[1]],
                hidden_states_ptb[:, encoder_hidden_states.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states_ptb = attn.to_add_out(encoder_hidden_states_ptb)
        
        if not attn.pre_only:
            hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
            hidden_states_ptb = attn.to_out[1](hidden_states_ptb)
        
        ################ Concatenate all three branches ###############
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])
        
        if is_joint_attn:
            encoder_hidden_states = torch.cat([encoder_hidden_states_org, encoder_hidden_states_ptb])
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

