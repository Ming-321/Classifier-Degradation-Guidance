"""
CDG Pipeline for Qwen-Image.

Uses Hook-based degradation on the standard QwenImagePipeline.
"""

from .pipeline import CDGQwenImagePipeline, CdgProcessIndexHook
from .utils import (
    PromptEncoding,
    build_tail_variant,
    decode_latents_to_pil,
    encode_postdrop_token_ids,
    encode_prompt_with_tokens,
    load_pipeline_split_devices,
)

__all__ = [
    "CDGQwenImagePipeline",
    "CdgProcessIndexHook",
    "PromptEncoding",
    "build_tail_variant",
    "decode_latents_to_pil",
    "encode_postdrop_token_ids",
    "encode_prompt_with_tokens",
    "load_pipeline_split_devices",
]
