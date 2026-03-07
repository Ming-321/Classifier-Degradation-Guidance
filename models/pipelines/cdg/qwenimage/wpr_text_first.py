"""
WPR (Weighted PageRank) token-importance for Qwen-Image, adapted to joint token order [text, image].

The original CDG implementation assumes joint order [image, text]. Qwen-Image's joint attention
order in diffusers is [text, image] (see diffusers/models/transformers/transformer_qwenimage.py).

This module keeps the WPR math the same, but changes the block slicing and "extract text scores"
indices to match [text, image].

Ported from Qwen-Image/wpr/wpr_text_first.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional, Tuple

import torch

CalScoreAlgorithm = Literal["WPR", "WPR_no_norm", "text_only_pagerank"]


def safe_normalize_rows(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise normalize with protection for all-zero rows."""
    row_sums = torch.sum(tensor, dim=-1, keepdim=True)
    out = tensor / (row_sums + eps)
    zero_mask = (row_sums < eps).squeeze(-1)
    if torch.any(zero_mask):
        out[zero_mask] = 0.0
    return out


def extract_attention_matrix_wpr_text_first(attn: torch.Tensor, text_seq_length: int) -> torch.Tensor:
    """
    WPR block-wise normalize and recombine for joint order [text, image].

    attn: (total_seq, total_seq), row = query, col = key.
    """
    total_seq_length = attn.shape[-1]
    if text_seq_length <= 0 or text_seq_length > total_seq_length:
        raise ValueError(f"Invalid text_seq_length={text_seq_length} for total_seq_length={total_seq_length}")
    image_seq_length = total_seq_length - text_seq_length
    if image_seq_length <= 0:
        raise ValueError(f"Image token count must be > 0, got {image_seq_length}")

    # Joint token order is [text, image]
    # Blocks:
    # - A_TT: text -> text
    # - A_TI: text -> image
    # - A_IT: image -> text
    # - A_II: image -> image
    A_TT = attn[:text_seq_length, :text_seq_length]
    A_TI = attn[:text_seq_length, text_seq_length:]
    A_IT = attn[text_seq_length:, :text_seq_length]
    A_II = attn[text_seq_length:, text_seq_length:]

    A_TT_norm = safe_normalize_rows(A_TT)
    A_TI_norm = safe_normalize_rows(A_TI)
    A_IT_norm = safe_normalize_rows(A_IT)
    A_II_norm = safe_normalize_rows(A_II)

    A = torch.zeros_like(attn)
    A[:text_seq_length, :text_seq_length] = A_TT_norm
    A[:text_seq_length, text_seq_length:] = A_TI_norm
    A[text_seq_length:, :text_seq_length] = A_IT_norm
    A[text_seq_length:, text_seq_length:] = A_II_norm

    return A


def extract_attention_matrix_text_only(attn: torch.Tensor, text_seq_length: int) -> torch.Tensor:
    """
    Extract and normalize the text-to-text block A_TT from a joint attention matrix with order [text, image].

    We re-normalize rows because the original attention rows sum to 1 over (text+image) keys.
    """
    total_seq_length = attn.shape[-1]
    if text_seq_length <= 0 or text_seq_length > total_seq_length:
        raise ValueError(f"Invalid text_seq_length={text_seq_length} for total_seq_length={total_seq_length}")
    A_TT = attn[:text_seq_length, :text_seq_length]
    return safe_normalize_rows(A_TT)


@dataclass(frozen=True)
class WPRConfig:
    epsilon: float = 1e-4
    max_iterations: int = 20
    norm_type: Literal["L1", "L2"] = "L1"
    damping_factor: float = 0.0  # 0.0 means undamped; 0.15 is standard


def variance_weighted_aggregate(
    scores_per_head: torch.Tensor,
    *,
    v_min: float = 0.0,
    v_max: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CDG-style variance-weighted aggregation across heads.

    This matches `classifier-degradation-guidance/models/pipelines/cdg/calculate_importance.py::variance_weighted`:
    - compute variance per head over tokens
    - keep heads with variance in [v_min, v_max]
    - final_scores = sqrt(sum(scores^2 * mask) / denom)

    Returns:
      final_scores: (text_seq_length,)
      variance_mask: (heads,) boolean tensor
    """
    if scores_per_head.dim() != 2:
        raise ValueError(f"scores_per_head must have shape (heads, text_seq_length), got {scores_per_head.shape}")

    head_variances = torch.var(scores_per_head, dim=1)  # (heads,)
    if v_max is None:
        v_max = float("inf")
    variance_mask = (head_variances >= v_min) & (head_variances <= v_max)

    squared_scores = scores_per_head**2
    masked_squared_scores = squared_scores * variance_mask.unsqueeze(1)
    denominator = variance_mask.sum().clamp(min=1.0)
    final_scores = torch.sqrt(masked_squared_scores.sum(dim=0) / denominator)

    return final_scores, variance_mask


def wpr_scores_text_first(
    attention_probs: torch.Tensor,
    text_seq_length: int,
    cfg: Optional[WPRConfig] = None,
    *,
    cal_score_algorithm: CalScoreAlgorithm = "WPR",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute WPR scores for text tokens given attention_probs with joint order [text, image].

    Args:
        attention_probs: (heads, total_seq, total_seq) or (1, heads, total_seq, total_seq)
        text_seq_length: number of text tokens (prefix of the joint sequence)
        cfg: WPRConfig

    Returns:
        scores_per_head: (heads, text_seq_length)
        scores_mean: (text_seq_length,)
    """
    if cfg is None:
        cfg = WPRConfig()

    if attention_probs.dim() == 4:
        attention_probs = attention_probs[0]
    if attention_probs.dim() != 3:
        raise ValueError(f"attention_probs must have shape (heads, S, S) or (1, heads, S, S), got {attention_probs.shape}")

    heads, total_seq_length, _ = attention_probs.shape
    if text_seq_length <= 0 or text_seq_length > total_seq_length:
        raise ValueError(f"Invalid text_seq_length={text_seq_length} for total_seq_length={total_seq_length}")

    device = attention_probs.device
    dtype = attention_probs.dtype

    scores = []
    for h in range(heads):
        curr = attention_probs[h]  # (S, S)
        if cal_score_algorithm == "WPR":
            A = extract_attention_matrix_wpr_text_first(curr, text_seq_length)
            adjacency = A.transpose(0, 1)
            matrix_size = total_seq_length
        elif cal_score_algorithm == "WPR_no_norm":
            # Raw joint attention. Note: attention_probs is already row-stochastic (softmax over keys).
            A = curr
            adjacency = A.transpose(0, 1)
            matrix_size = total_seq_length
        elif cal_score_algorithm == "text_only_pagerank":
            # SD3-style text-only PageRank on A_TT.
            A = extract_attention_matrix_text_only(curr, text_seq_length)
            adjacency = A.transpose(0, 1)
            matrix_size = text_seq_length
        else:
            raise ValueError(f"Unknown cal_score_algorithm={cal_score_algorithm}")

        s = torch.ones(matrix_size, device=device, dtype=dtype) / matrix_size
        if cfg.damping_factor > 0.0:
            teleport = torch.ones(matrix_size, device=device, dtype=dtype) / matrix_size

        for _ in range(cfg.max_iterations):
            s_prev = s
            if cfg.damping_factor > 0.0:
                s_voted = torch.matmul(adjacency, s_prev)
                s = (1 - cfg.damping_factor) * s_voted + cfg.damping_factor * teleport
            else:
                s = torch.matmul(adjacency, s_prev)

            if cfg.norm_type == "L1":
                norm_val = torch.sum(torch.abs(s))
            elif cfg.norm_type == "L2":
                norm_val = torch.linalg.norm(s, ord=2)
            else:
                raise ValueError(f"Unknown norm_type={cfg.norm_type}")

            eps_norm = 1e-9 if cfg.damping_factor > 0.0 else 1e-6
            s = s / (norm_val + eps_norm)

            if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
                if cfg.damping_factor > 0.0:
                    s = teleport
                break

            abs_diff = torch.sum(torch.abs(s - s_prev))
            if abs_diff < cfg.epsilon:
                break

        if cal_score_algorithm == "text_only_pagerank":
            scores.append(s)
        else:
            # Text is the prefix in [text, image]
            scores.append(s[:text_seq_length])

    scores_per_head = torch.stack(scores, dim=0)  # (heads, text_seq_length)
    scores_mean = torch.mean(scores_per_head, dim=0)
    return scores_per_head, scores_mean
