"""
Utilities to extract a joint attention probability matrix from Qwen-Image's MMDiT (diffusers).

We compute attention_probs manually from Q/K to feed token-importance algorithms (e.g. WPR).
This avoids changing diffusers internals (which use fused attention and do not expose probs).

Ported from Qwen-Image/wpr/qwenimage_attention.py.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen


@torch.no_grad()
def compute_joint_attention_probs_text_first(
    *,
    transformer,
    latents_packed: torch.Tensor,  # (B, img_seq, 64)
    prompt_embeds: torch.Tensor,  # (B, txt_seq, 3584)
    prompt_embeds_mask: Optional[torch.Tensor],  # (B, txt_seq)
    timestep: torch.Tensor,  # (B,) already scaled like pipeline (t/1000)
    img_shapes,
    txt_seq_lens,
    target_block: int,
    use_fp32: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Return attention_probs for the specified block.

    The joint token order is [text, image], consistent with diffusers' Qwen-Image implementation.

    Returns:
      attention_probs: (heads, total_seq, total_seq) on the current device
      text_seq_length: int
    """
    if timestep.dim() != 1:
        raise ValueError(f"timestep must be 1D (B,), got {timestep.shape}")

    # Mirror QwenImageTransformer2DModel.forward prelude.
    hidden_states = transformer.img_in(latents_packed)
    encoder_hidden_states = transformer.txt_norm(prompt_embeds)
    encoder_hidden_states = transformer.txt_in(encoder_hidden_states)

    temb = transformer.time_text_embed(timestep.to(hidden_states.dtype), hidden_states)
    image_rotary_emb = transformer.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

    # Advance blocks up to target_block-1 using the real block forward to keep states consistent.
    for idx, block in enumerate(transformer.transformer_blocks):
        if idx == target_block:
            break
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=prompt_embeds_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=None,
        )
    else:
        raise ValueError(f"target_block={target_block} out of range (num_layers={len(transformer.transformer_blocks)})")

    block = transformer.transformer_blocks[target_block]
    attn = block.attn

    # Mirror QwenImageTransformerBlock.forward up to the attention call.
    img_mod_params = block.img_mod(temb)
    txt_mod_params = block.txt_mod(temb)
    img_mod1, _img_mod2 = img_mod_params.chunk(2, dim=-1)
    txt_mod1, _txt_mod2 = txt_mod_params.chunk(2, dim=-1)

    img_normed = block.img_norm1(hidden_states)
    img_modulated, _img_gate1 = block._modulate(img_normed, img_mod1)

    txt_normed = block.txt_norm1(encoder_hidden_states)
    txt_modulated, _txt_gate1 = block._modulate(txt_normed, txt_mod1)

    # QKV projections (see QwenDoubleStreamAttnProcessor2_0).
    img_query = attn.to_q(img_modulated)
    img_key = attn.to_k(img_modulated)

    txt_query = attn.add_q_proj(txt_modulated)
    txt_key = attn.add_k_proj(txt_modulated)

    # [B, S, (H*D)] -> [B, S, H, D]
    img_query = img_query.unflatten(-1, (attn.heads, -1))
    img_key = img_key.unflatten(-1, (attn.heads, -1))
    txt_query = txt_query.unflatten(-1, (attn.heads, -1))
    txt_key = txt_key.unflatten(-1, (attn.heads, -1))

    # QK norm
    if attn.norm_q is not None:
        img_query = attn.norm_q(img_query)
    if attn.norm_k is not None:
        img_key = attn.norm_k(img_key)
    if attn.norm_added_q is not None:
        txt_query = attn.norm_added_q(txt_query)
    if attn.norm_added_k is not None:
        txt_key = attn.norm_added_k(txt_key)

    # RoPE (image_rotary_emb is a tuple (img_freqs, txt_freqs))
    if image_rotary_emb is not None:
        img_freqs, txt_freqs = image_rotary_emb
        img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
        img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
        txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
        txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

    # Joint order: [text, image] (matches diffusers)
    joint_query = torch.cat([txt_query, img_query], dim=1)
    joint_key = torch.cat([txt_key, img_key], dim=1)

    # Convert to [B, H, S, D]
    q = joint_query.permute(0, 2, 1, 3).contiguous()
    k = joint_key.permute(0, 2, 1, 3).contiguous()

    if use_fp32:
        qf = q.float()
        kf = k.float()
        scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(qf.shape[-1])
        probs = torch.softmax(scores, dim=-1)
        probs = probs.to(torch.float32)
    else:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        probs = torch.softmax(scores, dim=-1)

    # Batch size is expected to be 1 in our analysis script, but keep it generic.
    # Return shape (heads, S, S) for batch=1.
    if probs.shape[0] != 1:
        raise ValueError(f"Expected batch=1 for attention export, got batch={probs.shape[0]}")

    text_seq_length = txt_query.shape[1]
    return probs[0], int(text_seq_length)
