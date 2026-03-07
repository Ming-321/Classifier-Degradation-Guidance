"""
Module for calculating token importance using various algorithms including PageRank and attention-based methods.

This module provides unified interfaces for computing importance scores of text tokens
in diffusion models, supporting multiple algorithms such as PageRank variants and
cross-attention based approaches.
"""

import torch
import math
import torch.nn.functional as F
from typing import Optional, Literal
from enum import Enum


# Define algorithm mode enumeration
class PageRankMode(Enum):
    """
    Enumeration of PageRank algorithm modes for importance calculation.

    Each mode represents a different approach to applying PageRank on attention matrices:
    - FULL_NORMALIZED: Complete matrix with block-wise normalization (WPR)
    - FULL_RAW: Complete matrix without normalization (WPR_no_norm)
    - TEXT_ONLY: Text tokens only (text_only_pagerank)
    """

    FULL_NORMALIZED = "full_normalized"
    FULL_RAW = "full_raw"
    TEXT_ONLY = "text_only"


class NonPageRankMode(Enum):
    """
    Enumeration of non-PageRank algorithm modes for importance calculation.

    Currently supports:
    - CROSS_ATTENTION_IT: Cross-attention column sum method (cro_att_IT)
    """

    CROSS_ATTENTION_IT = "cross_attention_it"


class ImportanceCalculationError(ValueError):
    """
    Custom exception for importance calculation errors.

    Raised when there are issues with input validation, numerical instability,
    or other problems specific to importance calculation algorithms.
    """

    pass


def safe_normalize_rows(
    tensor: torch.Tensor, name: str = "", eps: float = 1e-6
) -> torch.Tensor:
    """
    Safely normalize tensor rows to avoid NaN values from zero rows.

    This function performs row-wise normalization while handling edge cases
    where entire rows might be zero, which would cause division by zero.

    Args:
        tensor (torch.Tensor): Input tensor to normalize
        name (str, optional): Tensor name for debugging purposes. Defaults to "".
        eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: Row-normalized tensor with same shape as input

    Note:
        Zero rows in the input will remain zero in the output.
    """
    row_sums = torch.sum(tensor, dim=-1, keepdim=True)
    normalized_tensor = tensor / (row_sums + eps)
    zero_mask = (row_sums < eps).squeeze(-1)
    if torch.any(zero_mask):
        normalized_tensor[zero_mask] = 0.0
    return normalized_tensor


def extract_attention_matrix(
    attention_probs: torch.Tensor, text_seq_length: int, mode: PageRankMode
) -> torch.Tensor:
    """
    Extract attention matrix according to the specified PageRank mode.

    This function processes the full attention matrix and extracts or transforms
    it based on the chosen PageRank algorithm mode.

    Args:
        attention_probs (torch.Tensor): Complete attention weight matrix
            with shape (total_tokens, total_tokens)
        text_seq_length (int): Number of text tokens
        mode (PageRankMode): PageRank algorithm mode

    Returns:
        torch.Tensor: Extracted/processed attention matrix

    Raises:
        ValueError: If unknown PageRank mode is specified

    Note:
        The attention matrix is assumed to have image tokens first,
        followed by text tokens.
    """
    total_seq_length = attention_probs.shape[-1]
    image_seq_length = total_seq_length - text_seq_length

    if mode == PageRankMode.FULL_NORMALIZED:
        # Block-wise normalization and recombination
        A_II = attention_probs[:image_seq_length, :image_seq_length]
        A_IT = attention_probs[:image_seq_length, image_seq_length:]
        A_TI = attention_probs[image_seq_length:, :image_seq_length]
        A_TT = attention_probs[image_seq_length:, image_seq_length:]

        # Normalize each sub-matrix
        A_II_norm = safe_normalize_rows(A_II, "A_II")
        A_IT_norm = safe_normalize_rows(A_IT, "A_IT")
        A_TI_norm = safe_normalize_rows(A_TI, "A_TI")
        A_TT_norm = safe_normalize_rows(A_TT, "A_TT")

        # Recombine normalized blocks
        A_combined = torch.zeros_like(attention_probs)
        A_combined[:image_seq_length, :image_seq_length] = A_II_norm
        A_combined[:image_seq_length, image_seq_length:] = A_IT_norm
        A_combined[image_seq_length:, :image_seq_length] = A_TI_norm
        A_combined[image_seq_length:, image_seq_length:] = A_TT_norm

        return A_combined

    elif mode == PageRankMode.FULL_RAW:
        # Use original matrix directly
        return attention_probs

    elif mode == PageRankMode.TEXT_ONLY:
        # Use only text portion and normalize
        if image_seq_length == 0:
            # Entire attention matrix is text-to-text
            A_TT = attention_probs
        else:
            # Extract text-to-text portion
            A_TT = attention_probs[image_seq_length:, image_seq_length:]
        return safe_normalize_rows(A_TT, "A_TT")

    else:
        raise ValueError(f"Unknown PageRank mode: {mode}")


def unified_pagerank(
    attention_probs: torch.Tensor,
    text_seq_length: int,
    mode: PageRankMode,
    epsilon: float = 1e-4,
    max_iterations: int = 20,
    norm_type: Literal["L1", "L2"] = "L1",
    damping_factor: float = 0.0,
) -> torch.Tensor:
    """
    Unified PageRank algorithm implementation for single batch processing.

    This function implements the PageRank algorithm with various configurations
    to compute importance scores for text tokens based on attention patterns.

    Args:
        attention_probs (torch.Tensor): Complete attention weight matrix
            with shape (heads, total_tokens, total_tokens)
        text_seq_length (int): Number of text tokens
        mode (PageRankMode): PageRank algorithm mode
        epsilon (float, optional): Convergence threshold. Defaults to 1e-4.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 20.
        norm_type (Literal['L1', 'L2'], optional): Normalization type. Defaults to 'L1'.
        damping_factor (float, optional): Damping factor for PageRank.
            0.0 means no damping, 0.15 is standard. Defaults to 0.0.

    Returns:
        torch.Tensor: Text token importance scores with shape (heads, text_seq_length)

    Raises:
        ImportanceCalculationError: If input validation fails or numerical issues occur

    Note:
        The algorithm processes each attention head separately and returns
        per-head importance scores for text tokens.
    """
    # Input validation
    if attention_probs.dim() != 3:
        raise ImportanceCalculationError(
            f"Input attention_probs must have shape (heads, total_tokens, total_tokens), "
            f"got {attention_probs.shape}"
        )

    h_dim, total_seq_length, _ = attention_probs.shape
    image_seq_length = total_seq_length - text_seq_length

    # For TEXT_ONLY mode, if no image tokens, use the entire attention matrix
    if mode == PageRankMode.TEXT_ONLY and image_seq_length == 0:
        # Entire attention matrix is text-to-text
        pass  # No need to check image_seq_length
    elif image_seq_length <= 0:
        raise ImportanceCalculationError(
            f"Image token count must be > 0, got {image_seq_length}"
        )

    device = attention_probs.device
    dtype = attention_probs.dtype

    head_results = []

    for head_idx in range(h_dim):
        curr_attention = attention_probs[head_idx]  # (total_tokens, total_tokens)

        # Extract attention matrix based on mode
        A_matrix = extract_attention_matrix(curr_attention, text_seq_length, mode)

        # Determine PageRank computation dimensions and target range
        if mode == PageRankMode.TEXT_ONLY:
            # Process only text tokens
            matrix_size = text_seq_length
            target_start_idx = 0
        else:
            # Process all tokens
            matrix_size = total_seq_length
            target_start_idx = image_seq_length

        # PageRank algorithm core implementation
        adjacency_matrix = A_matrix.transpose(0, 1)  # Transpose to get adjacency matrix

        # Initialize importance score vector
        s = torch.ones(matrix_size, device=device, dtype=dtype) / matrix_size

        # Prepare damping vector
        if damping_factor > 0.0:
            teleport_vector = (
                torch.ones(matrix_size, device=device, dtype=dtype) / matrix_size
            )

        # PageRank iterations
        for iteration_count in range(max_iterations):
            s_prev = s.clone()

            if damping_factor > 0.0:
                # Damped iteration
                s_voted = torch.matmul(adjacency_matrix, s_prev)
                s = (1 - damping_factor) * s_voted + damping_factor * teleport_vector
            else:
                # Undamped iteration
                s = torch.matmul(adjacency_matrix, s_prev)

            # Normalization
            if norm_type == "L1":
                norm_val = torch.sum(torch.abs(s))
            elif norm_type == "L2":
                norm_val = torch.linalg.norm(s, ord=2)
            else:
                raise ImportanceCalculationError(
                    f"Unknown normalization type: {norm_type}"
                )

            epsilon_norm = 1e-9 if damping_factor > 0.0 else 1e-6
            s = s / (norm_val + epsilon_norm)

            # Check numerical stability
            if torch.any(torch.isnan(s)) or torch.any(torch.isinf(s)):
                if damping_factor > 0.0:
                    print(f"Warning: NaN/Inf detected at iteration {iteration_count}!")
                    s = teleport_vector.clone()
                break

            # Check convergence
            abs_diff = torch.sum(torch.abs(s - s_prev))
            if abs_diff < epsilon:
                break

        # Extract text token importance scores
        if mode == PageRankMode.TEXT_ONLY:
            s_text = s  # Already text token scores
        else:
            s_text = s[target_start_idx:]  # Extract text portion

        head_results.append(s_text)

    # Stack all head results
    s_text_final = torch.stack(head_results, dim=0)  # (heads, text_seq_length)

    return s_text_final


def cross_attention_sum(
    attention_probs: torch.Tensor,
    text_seq_length: int,
) -> torch.Tensor:
    """
    Cross-attention sum algorithm for single batch processing.

    This algorithm computes text token importance by summing columns of the
    A_IT matrix (image tokens attending to text tokens).

    Args:
        attention_probs (torch.Tensor): Complete attention weight matrix
            with shape (heads, total_tokens, total_tokens)
        text_seq_length (int): Number of text tokens

    Returns:
        torch.Tensor: Text token importance scores with shape (heads, text_seq_length)

    Raises:
        ValueError: If input validation fails

    Note:
        This method focuses on how much image tokens attend to each text token,
        providing a cross-modal importance measure.
    """
    if attention_probs.dim() != 3:
        raise ValueError(
            f"Input attention_probs must have shape (heads, total_tokens, total_tokens), "
            f"got {attention_probs.shape}"
        )

    h_dim, total_seq_length, _ = attention_probs.shape
    image_seq_length = total_seq_length - text_seq_length

    if image_seq_length <= 0:
        raise ValueError(f"Image token count must be > 0, got {image_seq_length}")

    head_results = []

    for head_idx in range(h_dim):
        curr_attention = attention_probs[head_idx]  # (total_tokens, total_tokens)

        # Extract A_IT matrix: image tokens → text tokens
        A_IT = curr_attention[
            :image_seq_length, image_seq_length:
        ]  # (image_seq_length, text_seq_length)

        # Column-wise sum to get text token importance scores
        text_importance_scores = torch.sum(A_IT, dim=0)  # (text_seq_length,)

        head_results.append(text_importance_scores)

    # Stack all head results
    s_text_final = torch.stack(head_results, dim=0)  # (heads, text_seq_length)

    return s_text_final


# Calculate sorted indices based on scores


def variance_weighted(
    scores: torch.Tensor,
    process_type: str,
    v_min: float = 0.00,
    v_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute importance ranking using variance-weighted method.

    This method weights attention head contributions based on their score variance,
    filtering heads that fall within specified variance thresholds.

    Args:
        scores (torch.Tensor): Multi-head scores with shape (heads, text_seq_length)
        process_type (str): Sorting type ("min" or "max")
        v_min (float, optional): Minimum variance threshold. Defaults to 0.00.
        v_max (Optional[float], optional): Maximum variance threshold. Defaults to None.

    Returns:
        torch.Tensor: Sorted indices based on final aggregated scores

    Raises:
        ValueError: If unknown process_type is specified

    Note:
        The final score is computed as the square root of the sum of squared scores
        from heads that meet the variance criteria.
    """
    # Calculate variance for each attention head across all tokens
    head_variances = torch.var(scores, dim=1)  # Shape: (heads,)

    # Create mask for heads within variance threshold range
    if v_min is None:
        v_min = 0.00
    if v_max is None:
        v_max = float("inf")

    variance_mask = (v_min <= head_variances) & (
        head_variances <= v_max
    )  # Shape: (heads,)

    # Calculate squared scores
    squared_scores = scores**2  # Shape: (heads, text_seq_length)

    # Apply variance mask
    masked_squared_scores = squared_scores * variance_mask.unsqueeze(
        1
    )  # Broadcast to (heads, text_seq_length)

    # Calculate denominator: number of qualifying heads
    denominator = variance_mask.sum().clamp(min=1.0)  # Avoid division by zero

    # Calculate final scores: square root of sum of squares
    final_scores = torch.sqrt(masked_squared_scores.sum(dim=0) / denominator)

    # Sort based on process type
    if process_type == "min":
        final_sorted_indices = torch.argsort(final_scores, dim=-1, descending=False)
    elif process_type == "max":
        final_sorted_indices = torch.argsort(final_scores, dim=-1, descending=True)
    else:
        raise ValueError(f"Unknown process type: {process_type}")

    return final_sorted_indices, final_scores


def average_aggregation(scores: torch.Tensor, process_type: str) -> torch.Tensor:
    """
    Aggregate multi-head scores using simple averaging method.

    This method computes the average score across all attention heads
    and sorts tokens based on these averaged scores.

    Args:
        scores (torch.Tensor): Multi-head scores with shape (heads, text_seq_length)
        process_type (str): Sorting type ("min" or "max")

    Returns:
        torch.Tensor: Sorted indices based on averaged scores

    Raises:
        ValueError: If unknown process_type is specified
    """
    # Calculate average scores across heads
    average_scores = torch.mean(scores, dim=0)

    # Sort based on process type
    if process_type == "min":
        final_sorted_indices = torch.argsort(average_scores, dim=-1, descending=False)
    elif process_type == "max":
        final_sorted_indices = torch.argsort(average_scores, dim=-1, descending=True)
    else:
        raise ValueError(f"Unknown process type: {process_type}")

    return final_sorted_indices


class CalculateImportance:
    """
    Main class for calculating token importance using various algorithms.

    This class provides a unified interface for computing importance scores
    of text tokens in diffusion models, supporting multiple algorithms
    including PageRank variants and attention-based methods.

    Attributes:
        pipeline: The diffusion pipeline object
        calculate_params (dict): Configuration parameters for calculation algorithms
    """

    def __init__(self, pipeline, pipeline_name, calculate_params: dict):
        """
        Initialize the CalculateImportance instance.

        Args:
            pipeline: The diffusion pipeline object
            pipeline_name: The name of the pipeline
            calculate_params (dict): Configuration parameters containing algorithm settings
        """
        self.pipeline = pipeline
        self.pipeline_name = pipeline_name
        self.calculate_params = calculate_params

    def get_QKV(self, pipeline_args):
        """
        Extract Query, Key, and Value (QKV) matrices from the transformer pipeline.

        Args:
            pipeline: The diffusion pipeline object containing the transformer model
            pipeline_args (dict): Dictionary containing pipeline arguments including:
                - pipeline_name (str): Name of the pipeline (e.g., "sd3")
                - index_block (int): Index of the transformer block to extract from
                - hidden_states (torch.Tensor): Hidden states tensor
                - encoder_hidden_states (torch.Tensor): Encoder hidden states tensor
                - temb (torch.Tensor): Time embedding tensor

        Returns:
            tuple: A tuple containing (query, key, value, head_dim) where:
                - query (torch.Tensor): Query tensor with shape [batch_size, heads, seq_len, head_dim]
                - key (torch.Tensor): Key tensor with shape [batch_size, heads, seq_len, head_dim]
                - value (torch.Tensor): Value tensor with shape [batch_size, heads, seq_len, head_dim]
                - head_dim (int): Dimension of each attention head

        Raises:
            ValueError: If the pipeline_name is not supported
        """
        pipeline = self.pipeline
        if self.pipeline_name == "sd3":
            # Parse arguments
            transformer = pipeline.transformer
            block = transformer.transformer_blocks[pipeline_args["index_block"]]
            attn = block.attn
            hidden_states = pipeline_args["hidden_states"][1:2]
            encoder_hidden_states = pipeline_args["encoder_hidden_states"][1:2]
            temb = pipeline_args["temb"]
            batch_size = hidden_states.shape[0]

            # Normalize hidden_states
            if hasattr(block, "use_dual_attention") and block.use_dual_attention:
                (
                    hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    norm_hidden_states2,
                    gate_msa2,
                ) = block.norm1(hidden_states, emb=temb)
            else:
                hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
                    hidden_states, emb=temb
                )

            # Normalize encoder_hidden_states
            if hasattr(block, "context_pre_only") and block.context_pre_only:
                encoder_hidden_states = block.norm1_context(encoder_hidden_states, temb)
            else:
                (
                    encoder_hidden_states,
                    c_gate_msa,
                    c_shift_mlp,
                    c_scale_mlp,
                    c_gate_mlp,
                ) = block.norm1_context(encoder_hidden_states, emb=temb)

            # Get QKV matrices
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

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
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            return query, key, value, head_dim
        elif self.pipeline_name == "flux":
            # Parse arguments
            transformer = pipeline.transformer
            block = transformer.transformer_blocks[pipeline_args["index_block"]]
            attn = block.attn
            if (
                pipeline_args["hidden_states"].shape[0] == 1
                or pipeline_args["encoder_hidden_states"].shape[0] == 1
            ):
                print(
                    f"pipeline_args['hidden_states']: {pipeline_args['hidden_states'].shape}"
                )
                print(
                    f"pipeline_args['encoder_hidden_states']: {pipeline_args['encoder_hidden_states'].shape}"
                )
                raise ValueError(
                    "hidden_states.shape[0] == 1 or encoder_hidden_states.shape[0] == 1"
                )
            hidden_states = pipeline_args["hidden_states"][1:2]
            encoder_hidden_states = pipeline_args["encoder_hidden_states"][1:2]
            temb = pipeline_args["temb"]
            batch_size = hidden_states.shape[0]

            # Normalize hidden_states
            if hasattr(block, "use_dual_attention") and block.use_dual_attention:
                (
                    hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    norm_hidden_states2,
                    gate_msa2,
                ) = block.norm1(hidden_states, emb=temb)
            else:
                hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
                    hidden_states, emb=temb
                )

            # Normalize encoder_hidden_states
            if hasattr(block, "context_pre_only") and block.context_pre_only:
                encoder_hidden_states = block.norm1_context(encoder_hidden_states, temb)
            else:
                (
                    encoder_hidden_states,
                    c_gate_msa,
                    c_shift_mlp,
                    c_scale_mlp,
                    c_gate_mlp,
                ) = block.norm1_context(encoder_hidden_states, emb=temb)

            # Get QKV matrices
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

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
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            return query, key, value, head_dim
        else:
            raise ValueError(f"Unsupported pipeline: {pipeline_args['pipeline_name']}")

    def calculate_attention_probs(self, pipeline_args: dict) -> torch.Tensor:
        """
        Calculate attention probability matrices from pipeline arguments.

        This method extracts query, key, and value tensors from the pipeline
        and computes the softmax-normalized attention probabilities.

        Args:
            pipeline_args (dict): Pipeline arguments containing model states

        Returns:
            torch.Tensor: Attention probability matrix

        Note:
            The attention scores are computed using scaled dot-product attention
            with softmax normalization.
        """
        query, key, value, head_dim = self.get_QKV(pipeline_args)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            head_dim
        )
        attention_probs = F.softmax(attention_scores, dim=-1)

        return attention_probs

    def calculate_scores(
        self, attention_probs: torch.Tensor, text_seq_length: int
    ) -> torch.Tensor:
        """
        Unified interface for importance score calculation.

        This method routes to the appropriate algorithm based on configuration
        and computes importance scores for text tokens.

        Args:
            attention_probs (torch.Tensor): Attention weight matrix
            text_seq_length (int): Number of text tokens

        Returns:
            torch.Tensor: Importance scores for text tokens

        Raises:
            ValueError: If unknown algorithm is specified in configuration
        """
        # Get algorithm parameters from configuration
        algorithm = self.calculate_params.get(
            "cal_score_algorithm", "text_only_pagerank"
        )
        epsilon = self.calculate_params.get("epsilon", 1e-4)
        max_iterations = self.calculate_params.get("max_iterations", 20)
        norm_type = self.calculate_params.get("norm_type", "L1")
        damping_factor = self.calculate_params.get("damping_factor", 0.0)

        # Algorithm name to mode mapping
        algorithm_mapping = {
            "WPR": PageRankMode.FULL_NORMALIZED,
            "WPR_no_norm": PageRankMode.FULL_RAW,
            "text_only_pagerank": PageRankMode.TEXT_ONLY,
            "cro_att_IT": NonPageRankMode.CROSS_ATTENTION_IT,
        }

        if algorithm not in algorithm_mapping:
            raise ValueError(
                f"Unknown algorithm: {algorithm}, supported: {list(algorithm_mapping.keys())}"
            )

        # Handle input shape: support both 4D (batched) and 3D (single batch)
        if attention_probs.dim() == 4:
            # Take first batch
            attention_probs = attention_probs[0]
        elif attention_probs.dim() != 3:
            raise ValueError(
                f"Input attention_probs must have shape (batch_size, heads, total_tokens, total_tokens) "
                f"or (heads, total_tokens, total_tokens), got {attention_probs.shape}"
            )

        mode = algorithm_mapping[algorithm]

        if isinstance(mode, PageRankMode):
            # PageRank algorithm
            scores = unified_pagerank(
                attention_probs=attention_probs,
                text_seq_length=text_seq_length,
                mode=mode,
                epsilon=epsilon,
                max_iterations=max_iterations,
                norm_type=norm_type,
                damping_factor=damping_factor,
            )
        elif isinstance(mode, NonPageRankMode):
            # Non-PageRank algorithm
            if mode == NonPageRankMode.CROSS_ATTENTION_IT:
                scores = cross_attention_sum(
                    attention_probs=attention_probs,
                    text_seq_length=text_seq_length,
                )
        else:
            raise ValueError(f"Unimplemented algorithm mode: {mode}")

        return scores

    def calculate_sorted_indices(
        self, scores: torch.Tensor, process_type: str
    ) -> torch.Tensor:
        """
        Unified interface for computing sorted indices from importance scores.

        This method aggregates multi-head scores and returns sorted indices
        based on the configured aggregation algorithm.

        Args:
            scores (torch.Tensor): Multi-head scores with shape (heads, text_seq_length)
            process_type (str): Sorting type ("min" or "max")

        Returns:
            torch.Tensor: Sorted indices for token importance ranking

        Raises:
            ValueError: If unknown aggregation algorithm is specified
        """
        algorithm = self.calculate_params.get(
            "cal_sorted_indices_algorithm", "variance_weighted"
        )
        v_min = self.calculate_params.get("v_min", 0.00)
        v_max = self.calculate_params.get("v_max", None)

        if algorithm == "variance_weighted":
            return variance_weighted(scores, process_type, v_min, v_max)
        elif algorithm == "average":
            return average_aggregation(scores, process_type)
        else:
            raise ValueError(f"Unknown aggregation algorithm: {algorithm}")

    def __call__(self, pipeline_args: dict) -> torch.Tensor:
        """
        Main entry point for importance calculation.

        This method orchestrates the complete importance calculation pipeline:
        1. Extract text sequence length from pipeline arguments
        2. Calculate attention probabilities
        3. Compute importance scores
        4. Generate sorted indices

        Args:
            pipeline_args (dict): Pipeline arguments containing model states and metadata

        Returns:
            torch.Tensor: Sorted indices representing token importance ranking
        """
        # Get text_seq_length
        if "captured_layer_hidden_states" in pipeline_args:
            # From captured_layer_hidden_states
            text_seq_length = pipeline_args["captured_layer_hidden_states"].shape[1]
        elif "encoder_hidden_states" in pipeline_args:
            # Other methods: from encoder_hidden_states
            text_seq_length = pipeline_args["encoder_hidden_states"][1:2].shape[1]
        else:
            raise ValueError("Missing hidden states in pipeline_args")
        
        attention_probs = self.calculate_attention_probs(pipeline_args)
        scores = self.calculate_scores(attention_probs, text_seq_length)
        sorted_indices, final_scores = self.calculate_sorted_indices(
            scores, self.calculate_params["process_type"]
        )
        return sorted_indices, final_scores
