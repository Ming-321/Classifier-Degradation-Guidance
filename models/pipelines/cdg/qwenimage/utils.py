"""
Qwen-Image CDG utilities.

Ported from Qwen-Image/cdg_tail5_utils.py.
Provides prompt encoding, pipeline loading, VAE decoding, and tail variant building
for the Qwen-Image CDG pipeline.
"""

import os
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from diffusers import DiffusionPipeline


EXPECTED_TAIL5 = ["<|im_end|>", "Ċ", "<|im_start|>", "assistant", "Ċ"]


@dataclass
class PromptEncoding:
    prompt: str
    prompt_with_magic: str
    template_text: str
    drop_idx: int
    max_sequence_length: int
    token_ids: List[int]
    tokens: List[str]
    prompt_embeds: torch.Tensor  # [1, L, D] on text_device
    prompt_embeds_mask: torch.Tensor  # [1, L] on text_device

    @property
    def seq_len(self) -> int:
        return int(self.prompt_embeds.shape[1])


def resolve_path(model_path: str, maybe_relpath: str) -> str:
    if os.path.isabs(maybe_relpath):
        return maybe_relpath
    return os.path.join(model_path, maybe_relpath)


def load_prompts(prompts_file: str, prompts: Sequence[str]) -> List[str]:
    out: List[str] = []
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip("\n").strip()
                if s:
                    out.append(s)
    out.extend(list(prompts))
    return out


def load_pipeline_split_devices(
    model_path: str,
    transformer_device: torch.device,
    text_device: torch.device,
    vae_device: torch.device,
    torch_dtype: torch.dtype,
) -> Tuple[DiffusionPipeline, torch.nn.Module]:
    """
    Load once and pin components to devices:
      - transformer on transformer_device
      - text_encoder on text_device
      - vae on vae_device

    Returns (pipe, text_encoder_ref).
    """
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    pipe.transformer.to(transformer_device)
    pipe.text_encoder.to(text_device)
    pipe.vae.to(vae_device)

    # Force QwenImagePipeline._execution_device to be transformer_device.
    pipe.vae._hf_hook = types.SimpleNamespace(execution_device=str(transformer_device))

    return pipe, pipe.text_encoder


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def encode_prompt_with_tokens(
    pipe: DiffusionPipeline,
    text_encoder: torch.nn.Module,
    prompt: str,
    positive_magic: str,
    device: torch.device,
    max_sequence_length: int,
) -> PromptEncoding:
    """
    Equivalent to QwenImagePipeline._get_qwen_prompt_embeds + encode_prompt truncation,
    but without output_hidden_states=True (lower memory).
    """
    template = pipe.prompt_template_encode
    drop_idx = int(pipe.prompt_template_encode_start_idx)

    prompt_with_magic = prompt + (positive_magic or "")
    template_text = template.format(prompt_with_magic)

    # Keep tokenizer behavior consistent with diffusers pipeline.
    txt_tokens = pipe.tokenizer(
        [template_text],
        max_length=pipe.tokenizer_max_length + drop_idx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    enc_out = text_encoder.model(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        return_dict=True,
    )
    hidden_states = enc_out.last_hidden_state  # [1, S, D]

    split_hidden_states = pipe._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

    # Token ids/tokens aligned with split_hidden_states
    ids_all = txt_tokens.input_ids[0][txt_tokens.attention_mask[0].bool()].tolist()
    ids_all = ids_all[drop_idx:]
    tokens_all = pipe.tokenizer.convert_ids_to_tokens(ids_all)

    hs = split_hidden_states[0]  # [L, D], batch size=1
    if hs.shape[0] > max_sequence_length:
        hs = hs[:max_sequence_length]
        ids_all = ids_all[:max_sequence_length]
        tokens_all = tokens_all[:max_sequence_length]

    prompt_embeds = hs.unsqueeze(0).to(dtype=text_encoder.dtype, device=device)  # [1, L, D]
    prompt_embeds_mask = torch.ones((1, hs.shape[0]), dtype=torch.long, device=device)

    return PromptEncoding(
        prompt=prompt,
        prompt_with_magic=prompt_with_magic,
        template_text=template_text,
        drop_idx=drop_idx,
        max_sequence_length=max_sequence_length,
        token_ids=ids_all,
        tokens=tokens_all,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
    )


@torch.no_grad()
def encode_postdrop_token_ids(
    pipe: DiffusionPipeline,
    text_encoder: torch.nn.Module,
    token_ids_after_drop: Sequence[int],
    *,
    device: torch.device,
    template_prompt: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[str], int]:
    """
    Encode a *manually constructed* token-id sequence in the same "post-drop_idx" space used by QwenImagePipeline.

    This is useful when we need to control the final token sequence length (e.g. pad/align uncond to cond length).

    Returns:
      - prompt_embeds: [1, L, D]
      - prompt_embeds_mask: [1, L] (all ones)
      - token_ids (List[int]) of length L
      - tokens (List[str]) of length L
      - drop_idx (int)
    """
    drop_idx = int(pipe.prompt_template_encode_start_idx)
    template = pipe.prompt_template_encode

    # Get the first `drop_idx` token ids from the template prefix.
    template_text = template.format(template_prompt)
    tmp = pipe.tokenizer(
        [template_text],
        max_length=pipe.tokenizer_max_length + drop_idx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    prefix_ids = tmp.input_ids[0][tmp.attention_mask[0].bool()].tolist()[:drop_idx]
    if len(prefix_ids) != drop_idx:
        raise RuntimeError(f"Failed to extract template prefix ids: got {len(prefix_ids)} vs drop_idx={drop_idx}")

    ids = list(prefix_ids) + list(token_ids_after_drop)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    enc_out = text_encoder.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden_states = enc_out.last_hidden_state  # [1, drop_idx+L, D]
    hs = hidden_states[:, drop_idx:, :]  # [1, L, D]

    token_ids = list(token_ids_after_drop)
    tokens = pipe.tokenizer.convert_ids_to_tokens(token_ids)
    prompt_embeds = hs.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds_mask = torch.ones((1, len(token_ids)), dtype=torch.long, device=device)
    return prompt_embeds, prompt_embeds_mask, token_ids, tokens, drop_idx


def slice_tail_tokens(enc: PromptEncoding, tail_n: int = 5) -> Dict:
    L = enc.seq_len
    if L < tail_n:
        raise ValueError(f"seq_len={L} < tail_n={tail_n}")
    start = L - tail_n
    tail_ids = enc.token_ids[start:L]
    tail_tokens = enc.tokens[start:L]
    return {
        "tail_n": tail_n,
        "seq_len": L,
        "tail_start": start,
        "tail_end": L,
        "tail_token_ids": tail_ids,
        "tail_tokens": tail_tokens,
        "tail_tokens_match_expected": tail_tokens == EXPECTED_TAIL5,
        "expected_tail5": EXPECTED_TAIL5,
    }


def build_tail_variant(
    enc: PromptEncoding,
    tail_variant: str,
    tail_n: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Return (embeds, mask, meta) for using tail tokens as condition.

    tail_only:
      - embeds: [1, tail_n, D]
      - mask:   [1, tail_n] all ones
    tail_with_fill:
      - embeds: [1, L, D] with prefix filled by zeros, last tail_n positions = original tail embeddings
      - mask:   [1, L] all ones (to keep txt_seq_lens = L)
    """
    tail_meta = slice_tail_tokens(enc, tail_n=tail_n)
    L = enc.seq_len
    tail_embeds = enc.prompt_embeds[:, L - tail_n : L, :].contiguous()

    if tail_variant == "tail_only":
        embeds = tail_embeds
        mask = torch.ones((1, tail_n), dtype=torch.long, device=embeds.device)
        passed_tokens = tail_meta["tail_tokens"]
    elif tail_variant == "tail_with_fill":
        embeds = torch.zeros_like(enc.prompt_embeds)
        embeds[:, L - tail_n : L, :] = tail_embeds
        mask = torch.ones((1, L), dtype=torch.long, device=embeds.device)
        passed_tokens = ["<FILL>"] * (L - tail_n) + tail_meta["tail_tokens"]
    else:
        raise ValueError(f"Unknown tail_variant: {tail_variant}")

    meta = dict(tail_meta)
    meta.update(
        {
            "tail_variant": tail_variant,
            "passed_tokens": passed_tokens,
        }
    )
    return embeds, mask, meta


@torch.no_grad()
def generate_latents(
    pipe: DiffusionPipeline,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor],
    negative_prompt_embeds_mask: Optional[torch.Tensor],
    *,
    width: int,
    height: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Run denoising on transformer device and return packed latents.
    """
    out = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        generator=generator,
        output_type="latent",
    )
    return out.images


@torch.no_grad()
def decode_latents_to_pil(
    pipe: DiffusionPipeline,
    latents: torch.Tensor,
    *,
    vae_device: torch.device,
    width: int,
    height: int,
) -> "PIL.Image.Image":
    # Move latents to VAE device.
    latents = latents.to(vae_device, non_blocking=True)
    _sync(vae_device)

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)

    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean

    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    _sync(vae_device)

    image = image.detach().float().cpu()
    pil = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return pil


def embedding_stats(x: torch.Tensor) -> Dict[str, float]:
    """
    Simple, stable stats for logging/debugging (doesn't save full vectors).
    """
    with torch.no_grad():
        x_f = x.float()
        return {
            "shape": list(x.shape),
            "mean": float(x_f.mean().item()),
            "std": float(x_f.std().item()),
            "l2_norm": float(torch.linalg.vector_norm(x_f).item()),
        }
