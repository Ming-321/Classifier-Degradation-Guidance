"""
CDG Pipeline for Qwen-Image.

Implements SD3-style CDG using diffusers' Hook mechanism on the standard QwenImagePipeline.
For the uncond forward pass, a CdgProcessIndexHook degrades the text stream at a specified
transformer block by replacing prefix tokens with a filler token representation, while
keeping the tail_n positions unchanged to preserve global semantics.

Ported and refactored from Qwen-Image/experiment_cdg_qwenimage.py.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.hooks import HookRegistry, ModelHook
from diffusers.hooks.hooks import BaseState, StateManager
from PIL import Image

from .utils import (
    PromptEncoding,
    build_tail_variant,
    decode_latents_to_pil,
    encode_postdrop_token_ids,
    encode_prompt_with_tokens,
    load_pipeline_split_devices,
)


class _CDGHookState(BaseState):
    def reset(self, *args, **kwargs) -> None:
        return


class CdgProcessIndexHook(ModelHook):
    """
    SD3-style CDG for Qwen-Image without modifying diffusers' pipeline:
    - For uncond forward, we pass the SAME prompt embeds/mask as cond (so blocks < process_index see cond tokens).
    - At a specified transformer block index, we "degrade" the text stream by replacing the prefix tokens
      (first L - tail_n positions) with a filler token representation ("Ġ", token id=220), while keeping the tail_n
      positions unchanged to preserve global semantics.

    NOTE: We operate on the already-projected inner-dim text stream states (inside blocks),
    so we never change sequence length (RoPE is precomputed from txt_seq_lens).
    """

    _is_stateful = True

    def __init__(self, tail_n: int = 5, only_context: str = "uncond"):
        super().__init__()
        self._ctx = StateManager(_CDGHookState)
        self.tail_n = int(tail_n)
        self.only_context = str(only_context)
        self.active = False
        # Prefix replacement in *inner_dim* space (after transformer.txt_norm + transformer.txt_in).
        # Set by the caller per prompt before calling the pipeline.
        # Shape: [1, L, inner_dim], and we use [:, :L-tail_n, :].
        self.space220_inner: torch.Tensor | None = None

    def reset_state(self, module: torch.nn.Module):
        # Called by diffusers when pipelines reset stateful hooks after a forward pass.
        self.active = False
        self._ctx.reset()
        self.space220_inner = None
        return module

    def pre_forward(self, module, *args, **kwargs):
        # Only affect the uncond forward pass, and only when explicitly enabled by the caller.
        if not self.active:
            return args, kwargs
        if getattr(self._ctx, "_current_context", None) != self.only_context:
            return args, kwargs

        if "encoder_hidden_states" not in kwargs:
            return args, kwargs

        encoder_hidden_states = kwargs["encoder_hidden_states"]
        if encoder_hidden_states is None:
            return args, kwargs

        # [B, L, D]
        L = int(encoder_hidden_states.shape[1])
        tail_n = min(self.tail_n, L)
        prefix_len = L - tail_n
        if prefix_len <= 0:
            return args, kwargs

        if self.space220_inner is None:
            raise RuntimeError("CdgProcessIndexHook.space220_inner is not set. Set it before calling the pipeline.")

        fill = self.space220_inner.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        if fill.shape[0] != encoder_hidden_states.shape[0]:
            if fill.shape[0] == 1:
                fill = fill.expand(encoder_hidden_states.shape[0], -1, -1)
            else:
                raise RuntimeError(
                    f"space220_inner batch mismatch: B_fill={fill.shape[0]} vs current_B={encoder_hidden_states.shape[0]}"
                )
        if int(fill.shape[1]) < prefix_len:
            raise RuntimeError(f"space220_inner too short: L_fill={int(fill.shape[1])} < prefix_len={prefix_len}")

        out = encoder_hidden_states.clone()
        out[:, :prefix_len, :] = fill[:, :prefix_len, :]
        kwargs["encoder_hidden_states"] = out

        return args, kwargs


class CDGQwenImagePipeline:
    """
    Wrapper class that manages the Qwen-Image CDG generation flow.

    This is not a diffusers Pipeline subclass — it wraps the standard QwenImagePipeline
    and uses hooks for CDG degradation.
    """

    def __init__(
        self,
        pipe: DiffusionPipeline,
        text_encoder: torch.nn.Module,
        transformer_device: torch.device,
        text_device: torch.device,
        vae_device: torch.device,
    ):
        self.pipe = pipe
        self.text_encoder = text_encoder
        self.transformer_device = transformer_device
        self.text_device = text_device
        self.vae_device = vae_device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        transformer_gpu: int = 0,
        text_encoder_gpu: int = 1,
        vae_gpu: int = 2,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "CDGQwenImagePipeline":
        """Load the pipeline and distribute components across GPUs."""
        transformer_device = torch.device(f"cuda:{transformer_gpu}")
        text_device = torch.device(f"cuda:{text_encoder_gpu}")
        vae_device = torch.device(f"cuda:{vae_gpu}")

        pipe, text_encoder = load_pipeline_split_devices(
            model_path=model_path,
            transformer_device=transformer_device,
            text_device=text_device,
            vae_device=vae_device,
            torch_dtype=torch_dtype,
        )
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        return cls(
            pipe=pipe,
            text_encoder=text_encoder,
            transformer_device=transformer_device,
            text_device=text_device,
            vae_device=vae_device,
        )

    def generate_cdg(
        self,
        prompt: str,
        *,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        true_cfg_scale: float = 4.0,
        tail_n: int = 5,
        tail_variant: str = "tail_only",
        process_index: int = -1,
        positive_magic: str = "",
        max_sequence_length: int = 512,
    ) -> Image.Image:
        """
        Generate a single image using CDG.

        Args:
            prompt: Text prompt.
            seed: Random seed.
            width/height: Image dimensions.
            num_inference_steps: Denoising steps.
            true_cfg_scale: CFG scale for Qwen-Image.
            tail_n: Number of tail tokens to preserve.
            tail_variant: "tail_only" or "tail_with_fill".
            process_index: Transformer block index for hook-based CDG. -1 means no hook (use tail directly).
            positive_magic: Optional suffix appended to prompt.
            max_sequence_length: Max token sequence length.

        Returns:
            PIL Image.
        """
        # 1) Encode prompt
        full_enc = encode_prompt_with_tokens(
            pipe=self.pipe,
            text_encoder=self.text_encoder,
            prompt=prompt,
            positive_magic=positive_magic,
            device=self.text_device,
            max_sequence_length=max_sequence_length,
        )
        _, _, tail_meta = build_tail_variant(full_enc, tail_variant=tail_variant, tail_n=tail_n)

        # 2) Prepare tensors on transformer device
        full_embeds = full_enc.prompt_embeds.to(self.transformer_device, non_blocking=True)
        full_mask = full_enc.prompt_embeds_mask.to(self.transformer_device, non_blocking=True)

        gen = torch.Generator(device=self.transformer_device).manual_seed(seed)

        use_hook = process_index is not None and int(process_index) >= 0

        if not use_hook:
            # CDG (no process_index): directly use cond tail as uncond condition.
            if full_enc.seq_len < tail_n:
                raise ValueError(f"cond seq_len={full_enc.seq_len} < tail_n={tail_n}")
            cond_tail = full_enc.prompt_embeds[:, full_enc.seq_len - tail_n : full_enc.seq_len, :]
            neg_swapped = cond_tail.to(self.transformer_device, non_blocking=True)
            neg_swapped_mask = torch.ones(
                (neg_swapped.shape[0], neg_swapped.shape[1]), dtype=torch.long, device=neg_swapped.device
            )
            latents = self.pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=full_embeds,
                prompt_embeds_mask=full_mask,
                negative_prompt_embeds=neg_swapped,
                negative_prompt_embeds_mask=neg_swapped_mask,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=gen,
                output_type="latent",
            ).images
        else:
            # SD3-style CDG with process_index hook
            process_index = int(process_index)
            num_blocks = len(self.pipe.transformer.transformer_blocks)
            if process_index >= num_blocks:
                raise ValueError(f"process_index={process_index} out of range for num_blocks={num_blocks}")

            # Register hook
            cdg_hook = CdgProcessIndexHook(tail_n=tail_n, only_context="uncond")
            HookRegistry.check_if_exists_or_initialize(
                self.pipe.transformer.transformer_blocks[process_index]
            ).register_hook(cdg_hook, name="cdg_process_index_hook")

            # Prepare space220 inner representation
            L = int(full_enc.seq_len)
            space_ids = [220] * L
            space_text, _, _, _, _ = encode_postdrop_token_ids(
                pipe=self.pipe,
                text_encoder=self.text_encoder,
                token_ids_after_drop=space_ids,
                device=self.text_device,
                template_prompt="",
            )
            space_inner = self.pipe.transformer.txt_in(
                self.pipe.transformer.txt_norm(space_text.to(self.transformer_device, non_blocking=True))
            )
            cdg_hook.space220_inner = space_inner
            cdg_hook.active = True

            try:
                latents = self.pipe(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=full_embeds,
                    prompt_embeds_mask=full_mask,
                    negative_prompt_embeds=full_embeds,
                    negative_prompt_embeds_mask=full_mask,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=gen,
                    output_type="latent",
                ).images
            finally:
                cdg_hook.active = False

        # 3) Decode
        image = decode_latents_to_pil(
            self.pipe, latents, vae_device=self.vae_device, width=width, height=height
        )
        return image

    def generate_cfg_baseline(
        self,
        prompt: str,
        *,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        positive_magic: str = "",
        max_sequence_length: int = 512,
    ) -> Image.Image:
        """Generate a single image using standard CFG (baseline)."""
        # Encode cond
        full_enc = encode_prompt_with_tokens(
            pipe=self.pipe,
            text_encoder=self.text_encoder,
            prompt=prompt,
            positive_magic=positive_magic,
            device=self.text_device,
            max_sequence_length=max_sequence_length,
        )
        full_embeds = full_enc.prompt_embeds.to(self.transformer_device, non_blocking=True)
        full_mask = full_enc.prompt_embeds_mask.to(self.transformer_device, non_blocking=True)

        # Encode uncond (negative prompt)
        neg_enc = encode_prompt_with_tokens(
            pipe=self.pipe,
            text_encoder=self.text_encoder,
            prompt=negative_prompt,
            positive_magic="",
            device=self.text_device,
            max_sequence_length=max_sequence_length,
        )
        neg_embeds = neg_enc.prompt_embeds.to(self.transformer_device, non_blocking=True)
        neg_mask = neg_enc.prompt_embeds_mask.to(self.transformer_device, non_blocking=True)

        gen = torch.Generator(device=self.transformer_device).manual_seed(seed)
        latents = self.pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=full_embeds,
            prompt_embeds_mask=full_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_embeds_mask=neg_mask,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=gen,
            output_type="latent",
        ).images

        image = decode_latents_to_pil(
            self.pipe, latents, vae_device=self.vae_device, width=width, height=height
        )
        return image
