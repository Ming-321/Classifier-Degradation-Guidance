from typing import List, Dict, Tuple, Optional, Any
from .calculate_importance import CalculateImportance
import torch
import datetime
import os
import json
import numpy as np
import matplotlib.pyplot as plt


# Configure matplotlib font for better international character support
try:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "WenQuanYi Zen Hei"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    # Use default font if international fonts are not available
    pass

import random


class ProcessToken:
    """
    Process tokens based on their content and padding status, i.e., interpolate with tokens encoded from an empty string. To be compatible with new models, theoretically, only the code for analyzing content and padding tokens needs to be modified.
    A class for processing tokens in the context of guided generation pipelines.

    This class handles token importance calculation, content/padding analysis, and selective
    token processing based on configurable parameters. It supports both CLIP and T5
    token processing with customizable keep ratios.

    Attributes:
        pipeline: The diffusion pipeline instance
        process_params (Dict[str, Any]): Parameters for token processing
        prompt (str): Input prompt text
        context_embedder: Context embedding layer from the transformer
        empty_encoder_hidden_states: Empty encoder hidden states tensor
        debug (bool): Whether to enable debug mode
        all_content_flags (List[bool]): Token content/padding analysis results
        all_tokens (List[str]): Token text strings (debug mode only)
        importance_calculator: Instance for calculating token importance
        device: Torch device for computations
        debug_data (Dict): Debug data collection (debug mode only)
    """

    def __init__(
        self,
        pipeline,
        pipeline_name,
        prompt: str,
        empty_encoder_hidden_states: torch.Tensor,
        process_params: Dict[str, Any],
        device: torch.device,
        debug: bool = False,
        captured_layer_hidden_states: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the ProcessToken instance.

        Args:
            pipeline: The diffusion pipeline instance
            pipeline_name: The name of the pipeline
            prompt (str): Input prompt text for token processing
            empty_encoder_hidden_states (torch.Tensor): Empty encoder hidden states
            process_params (Dict[str, Any]): Processing parameters including:
                - calculate_params: Parameters for importance calculation
                - keep_ratio: Token keep ratios configuration
                - separate_clip_t5: Whether to process CLIP and T5 separately
            device (torch.device): Device for tensor operations
            debug (bool, optional): Enable debug mode. Defaults to False.
            captured_layer_hidden_states (Optional[torch.Tensor], optional): Captured layer hidden states from hook. Defaults to None.
        """
        self.pipeline = pipeline
        self.pipeline_name = pipeline_name
        self.process_params = process_params
        self.prompt = prompt
        if self.pipeline_name in ["sd3", "flux"]:
            self.context_embedder = self.pipeline.transformer.context_embedder
            # Ensure both context_embedder and empty_encoder_hidden_states are on the same device
            if empty_encoder_hidden_states.device != device:
                empty_encoder_hidden_states = empty_encoder_hidden_states.to(device)

            # Also ensure context_embedder is on the correct device
            self.context_embedder = self.context_embedder.to(device)

            self.empty_encoder_hidden_states = self.context_embedder(
                empty_encoder_hidden_states
            )
        else:
            self.empty_encoder_hidden_states = empty_encoder_hidden_states
        self.debug = debug
        
        # Store hidden states captured by hook
        self.captured_layer_hidden_states = captured_layer_hidden_states

        # Analyze token content/padding status and get tokens if in debug mode
        if debug:
            self.all_content_flags, self.all_tokens = self.analyze_token_content_status(
                prompt, return_tokens=True
            )
        else:
            self.all_content_flags, _ = self.analyze_token_content_status(
                prompt, return_tokens=False
            )
            self.all_tokens = None

        if self.debug:
            print(f"all_content_flags: {self.all_content_flags}")

        self.importance_calculator = CalculateImportance(
            pipeline,
            pipeline_name,
            process_params["calculate_params"],
        )
        self.device = device

        if process_params.get("all_use_first_step_importance"):
            self.first_step_importance = None

        # Initialize random generator for random_sorted_indices
        # If random_index_seed is set, use independent Random instance
        # Otherwise, use global random module (backward compatible)
        random_index_seed = process_params.get("random_index_seed", None)
        if random_index_seed is not None:
            self._index_random = random.Random(random_index_seed)
        else:
            self._index_random = None

        # Initialize debug data collection if debug mode is enabled
        if self.debug:
            self.debug_data = {
                "metadata": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "pipeline_name": self.pipeline_name,
                    "prompt": self.prompt,
                    "process_params": self.process_params,
                    "tokenization_details": self.all_tokens,
                    "token_content_flags": self.all_content_flags,
                },
                "importance_records": [],
            }

    def get_keep_mask(
        self, sorted_indices: List[int], separate_clip_t5: bool, ratio: Dict[str, float]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Generate a keep mask for token selection based on importance ranking and specified ratios.

        This method creates a binary mask indicating which tokens to keep from the
        original encoder hidden states versus using empty/negative states, based on
        the importance ranking provided by sorted_indices.

        Args:
            sorted_indices (List[int]): Token indices sorted by importance. The actual order
                                      depends on the process_type parameter used in calculation:
                                      - "min": least important tokens first
                                      - "max": most important tokens first
            separate_clip_t5 (bool): Whether to process CLIP and T5 tokens separately
            ratio (Dict[str, float]): Keep ratio configuration. Can be either:
                - {"all": k}: Keep ratio k for all tokens
                - {"content": k1, "padding": k2}: Different ratios for content/padding tokens

        Returns:
            Tuple[torch.Tensor, List[int]]: A tuple containing:
                - keep_mask (torch.Tensor): Binary mask for token selection
                - kept_indices (List[int]): List of indices for kept tokens

        Note:
            The method always selects the first X% of tokens from the sorted_indices list,
            where X is determined by the ratio. Whether these are the most or least important
            tokens depends on the process_type used during importance calculation.
            
            The degrade_target parameter (from process_params) controls which encoder tokens to degrade:
            - "all": Degrade all tokens (default, backward compatible)
            - "clip_only": Only degrade CLIP tokens (0-76), T5 tokens (77+) use null/empty embeddings
            - "t5_only": Only degrade T5 tokens (77+), CLIP tokens (0-76) use null/empty embeddings
            
            Note: For "clip_only" and "t5_only", the non-target region uses null embeddings (keep_mask=0),
            NOT the original embeddings. This isolates the effect of degrading specific encoders.
        """
        seq_length = len(self.all_content_flags)
        device = self.device
        clip_length = 77  # SD3 CLIP token length

        # Get target dtype from empty_encoder_hidden_states for consistency
        target_dtype = self.empty_encoder_hidden_states.dtype

        # Initialize keep_mask as all zeros (use null/empty embeddings by default)
        keep_mask = torch.zeros(seq_length, device=device, dtype=target_dtype)
        
        # Get degrade_target parameter: "all", "clip_only", or "t5_only"
        degrade_target = self.process_params.get("degrade_target", "all")
        
        # Note: Non-target regions stay at keep_mask=0 (null embeddings)
        # Only the target region will have ratio-based degradation applied below

        # Get token content/padding information
        content_mask = torch.tensor(self.all_content_flags, dtype=torch.bool, device=device)

        # Convert sorted_indices to tensor if needed for consistent processing
        if isinstance(sorted_indices, list):
            sorted_indices = torch.tensor(
                sorted_indices, device=device, dtype=torch.long
            )
        elif not isinstance(sorted_indices, torch.Tensor):
            sorted_indices = torch.tensor(
                sorted_indices, device=device, dtype=torch.long
            )

        kept_indices = []  # Track indices of kept tokens

        if "all" in ratio:
            # Keep tokens by ratio for all tokens uniformly, using importance ranking
            all_ratio = ratio["all"]

            if separate_clip_t5:
                # Separate CLIP and T5 tokens from sorted indices
                clip_sorted = sorted_indices[sorted_indices < clip_length]
                t5_sorted = sorted_indices[sorted_indices >= clip_length]

                # CLIP part processing - skip if degrade_target is "t5_only"
                if degrade_target != "t5_only":
                    clip_keep_count = int(clip_length * all_ratio)
                    if clip_keep_count > 0 and len(clip_sorted) > 0:
                        clip_to_keep = clip_sorted[: min(clip_keep_count, len(clip_sorted))]
                        keep_mask[clip_to_keep] = 1.0
                        kept_indices.extend(
                            clip_to_keep.cpu().tolist()
                            if torch.is_tensor(clip_to_keep)
                            else clip_to_keep
                        )

                # T5 part processing - skip if degrade_target is "clip_only"
                if degrade_target != "clip_only":
                    t5_length = seq_length - clip_length
                    t5_keep_count = int(t5_length * all_ratio)
                    if t5_keep_count > 0 and len(t5_sorted) > 0:
                        t5_to_keep = t5_sorted[: min(t5_keep_count, len(t5_sorted))]
                        keep_mask[t5_to_keep] = 1.0
                        kept_indices.extend(
                            t5_to_keep.cpu().tolist()
                            if torch.is_tensor(t5_to_keep)
                            else t5_to_keep
                        )
            else:
                # Process all tokens uniformly using importance ranking
                # Note: When degrade_target is not "all", filter to target range
                if degrade_target == "clip_only":
                    target_indices = sorted_indices[sorted_indices < clip_length]
                    target_length = clip_length
                elif degrade_target == "t5_only":
                    target_indices = sorted_indices[sorted_indices >= clip_length]
                    target_length = seq_length - clip_length
                else:
                    target_indices = sorted_indices
                    target_length = seq_length
                
                keep_count = int(target_length * all_ratio)
                if keep_count > 0:
                    # keep first X% from sorted order
                    tokens_to_keep = target_indices[
                        : min(keep_count, len(target_indices))
                    ]
                    keep_mask[tokens_to_keep] = 1.0
                    kept_indices.extend(
                        tokens_to_keep.cpu().tolist()
                        if torch.is_tensor(tokens_to_keep)
                        else tokens_to_keep
                    )
        else:
            # Process content and padding tokens with different ratios, using importance ranking
            content_ratio = ratio.get("content", 0.0)
            padding_ratio = ratio.get("padding", 0.0)

            if separate_clip_t5:
                # Separate CLIP and T5 tokens from sorted indices
                clip_sorted = sorted_indices[sorted_indices < clip_length]
                t5_sorted = sorted_indices[sorted_indices >= clip_length]

                # Process CLIP part - skip if degrade_target is "t5_only"
                if degrade_target != "t5_only":
                    # Separate content and padding CLIP tokens by importance
                    clip_content_sorted = clip_sorted[content_mask[clip_sorted]]
                    clip_padding_sorted = clip_sorted[~content_mask[clip_sorted]]

                    # CLIP content tokens - keep first X% from sorted order
                    clip_content_keep_count = int(len(clip_content_sorted) * content_ratio)
                    if clip_content_keep_count > 0:
                        clip_content_to_keep = clip_content_sorted[:clip_content_keep_count]
                        keep_mask[clip_content_to_keep] = 1.0
                        kept_indices.extend(
                            clip_content_to_keep.cpu().tolist()
                            if torch.is_tensor(clip_content_to_keep)
                            else clip_content_to_keep
                        )

                    # CLIP padding tokens - keep first X% from sorted order
                    clip_padding_keep_count = int(len(clip_padding_sorted) * padding_ratio)
                    if clip_padding_keep_count > 0:
                        clip_padding_to_keep = clip_padding_sorted[:clip_padding_keep_count]
                        keep_mask[clip_padding_to_keep] = 1.0
                        kept_indices.extend(
                            clip_padding_to_keep.cpu().tolist()
                            if torch.is_tensor(clip_padding_to_keep)
                            else clip_padding_to_keep
                        )

                # Process T5 part - skip if degrade_target is "clip_only"
                if degrade_target != "clip_only":
                    # Separate content and padding T5 tokens by importance
                    t5_content_sorted = t5_sorted[content_mask[t5_sorted]]
                    t5_padding_sorted = t5_sorted[~content_mask[t5_sorted]]

                    # T5 content tokens - keep first X% from sorted order
                    t5_content_keep_count = int(len(t5_content_sorted) * content_ratio)
                    if t5_content_keep_count > 0:
                        t5_content_to_keep = t5_content_sorted[:t5_content_keep_count]
                        keep_mask[t5_content_to_keep] = 1.0
                        kept_indices.extend(
                            t5_content_to_keep.cpu().tolist()
                            if torch.is_tensor(t5_content_to_keep)
                            else t5_content_to_keep
                        )

                    # T5 padding tokens - keep first X% from sorted order
                    t5_padding_keep_count = int(len(t5_padding_sorted) * padding_ratio)
                    if t5_padding_keep_count > 0:
                        t5_padding_to_keep = t5_padding_sorted[:t5_padding_keep_count]
                        keep_mask[t5_padding_to_keep] = 1.0
                        kept_indices.extend(
                            t5_padding_to_keep.cpu().tolist()
                            if torch.is_tensor(t5_padding_to_keep)
                            else t5_padding_to_keep
                        )
            else:
                # Process without CLIP/T5 separation
                # Note: When degrade_target is not "all", filter sorted_indices to target range
                if degrade_target == "clip_only":
                    # Only process CLIP tokens (0-76)
                    sorted_indices = sorted_indices[sorted_indices < clip_length]
                elif degrade_target == "t5_only":
                    # Only process T5 tokens (77+)
                    sorted_indices = sorted_indices[sorted_indices >= clip_length]
                
                # Separate content and padding tokens by importance
                content_sorted = sorted_indices[content_mask[sorted_indices]]
                padding_sorted = sorted_indices[~content_mask[sorted_indices]]

                # content tokens - keep first X% from sorted order
                content_keep_count = int(len(content_sorted) * content_ratio)
                if content_keep_count > 0:
                    content_to_keep = content_sorted[:content_keep_count]
                    keep_mask[content_to_keep] = 1.0
                    kept_indices.extend(
                        content_to_keep.cpu().tolist()
                        if torch.is_tensor(content_to_keep)
                        else content_to_keep
                    )

                # padding tokens - keep first X% from sorted order
                padding_keep_count = int(len(padding_sorted) * padding_ratio)
                if padding_keep_count > 0:
                    padding_to_keep = padding_sorted[:padding_keep_count]
                    keep_mask[padding_to_keep] = 1.0
                    kept_indices.extend(
                        padding_to_keep.cpu().tolist()
                        if torch.is_tensor(padding_to_keep)
                        else padding_to_keep
                    )

        return keep_mask, kept_indices

    def analyze_token_content_status(
        self, prompt: str, return_tokens: bool = False
    ) -> Tuple[List[bool], Optional[List[str]]]:
        """
        Analyze the content status of text-encoded tokens; the encoder configuration varies by model, so analysis must be based on model configuration.
        Analyze token content status to identify padding tokens.

        This method processes both CLIP and T5 tokens to determine which tokens
        are actual content tokens versus padding tokens that should be handled
        differently during processing.

        Args:
            prompt (str): Input prompt text to analyze
            return_tokens (bool): Whether to return token strings (for debug mode)

        Returns:
            Tuple[List[bool], Optional[List[str]]]: Token content flags list and optionally token strings
                       where True indicates content tokens and False indicates padding tokens.
        """
        all_content_flags = []
        all_tokens = [] if return_tokens else None

        if self.pipeline_name == "sd3":
            # SD3 model has three text encoders (tokenizer, tokenizer_2, tokenizer_3). Their encoded results are concatenated to form the final text encoding, so both CLIP (tokenizer) and T5 (tokenizer_2) need to be considered in content/padding analysis.
            # Get CLIP tokenizer (first 77 tokens)
            clip_tokenizer = self.pipeline.tokenizer
            clip_inputs = clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            # Analyze CLIP token content status
            clip_content_flags = []
            first_endoftext_seen = False  # Track first <|endoftext|> token

            for i in range(77):
                if i < len(clip_inputs.input_ids[0]):
                    token_id = clip_inputs.input_ids[0][i].item()
                    token_text = clip_tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    ).strip()

                    if return_tokens:
                        all_tokens.append(token_text)

                    # Check for padding tokens
                    if token_text == "<|endoftext|>":
                        if not first_endoftext_seen:
                            # First <|endoftext|> is sentence end marker (content)
                            is_content = True
                            first_endoftext_seen = True
                        else:
                            # Subsequent <|endoftext|> tokens are padding
                            is_content = False
                    else:
                        # All other tokens are content tokens
                        is_content = True

                    clip_content_flags.append(is_content)
                else:
                    clip_content_flags.append(False)  # Out of range tokens are padding
                    if return_tokens:
                        all_tokens.append("<PAD>")

            # Get T5 tokenizer - SD3 has tokenizer_3, FLUX only has tokenizer_2
            # SD3 model uses tokenizer_3 for T5
            t5_tokenizer = self.pipeline.tokenizer_3
            max_length = 256

            # Analyze T5 token content status
            t5_inputs = t5_tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            t5_content_flags = []
            for i in range(max_length):
                if i < len(t5_inputs.input_ids[0]):
                    token_id = t5_inputs.input_ids[0][i].item()
                    token_text = t5_tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    ).strip()

                    if return_tokens:
                        all_tokens.append(token_text)

                    # Check for T5 padding tokens
                    is_content = token_text not in ["<pad>"]
                    t5_content_flags.append(is_content)
                else:
                    t5_content_flags.append(False)  # Out of range tokens are padding
                    if return_tokens:
                        all_tokens.append("<PAD>")

            # Combine CLIP and T5 validity information (total 333 tokens)
            all_content_flags = clip_content_flags + t5_content_flags

        elif self.pipeline_name == "flux":
            # Although the Flux model also uses two encoders, CLIP and T5, CLIP is only used for pooled_prompt_embeds, and prompt_embeds are fully encoded by T5. Therefore, content/padding analysis only considers T5.
            t5_tokenizer = self.pipeline.tokenizer_2
            max_length = 512
            # Analyze T5 token content status
            t5_inputs = t5_tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            t5_content_flags = []
            for i in range(max_length):
                if i < len(t5_inputs.input_ids[0]):
                    token_id = t5_inputs.input_ids[0][i].item()
                    token_text = t5_tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    ).strip()

                    if return_tokens:
                        all_tokens.append(token_text)

                    # Check for T5 padding tokens
                    is_content = token_text not in ["<pad>"]
                    t5_content_flags.append(is_content)
                else:
                    t5_content_flags.append(False)  # Out of range tokens are padding
                    if return_tokens:
                        all_tokens.append("<PAD>")
            all_content_flags = t5_content_flags

        elif self.pipeline_name == "sd":
            clip_tokenizer = self.pipeline.tokenizer
            clip_inputs = clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            clip_content_flags = []
            first_endoftext_seen = False  # Track first <|endoftext|> token

            for i in range(77):
                if i < len(clip_inputs.input_ids[0]):
                    token_id = clip_inputs.input_ids[0][i].item()
                    token_text = clip_tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    ).strip()

                    if return_tokens:
                        all_tokens.append(token_text)

                    # Check for padding tokens
                    if token_text == "<|endoftext|>":
                        if not first_endoftext_seen:
                            # First <|endoftext|> is sentence end marker (content)
                            is_content = True
                            first_endoftext_seen = True
                        else:
                            # Subsequent <|endoftext|> tokens are padding
                            is_content = False
                    else:
                        # All other tokens are content tokens
                        is_content = True

                    clip_content_flags.append(is_content)
                else:
                    clip_content_flags.append(False)  # Out of range tokens are padding
                    if return_tokens:
                        all_tokens.append("<PAD>")
            all_content_flags = clip_content_flags
        else:
            raise ValueError(
                f"Unsupported pipeline '{self.pipeline_name}' for CDG token processing. "
                f"Supported pipelines: sd3, flux, sd"
            )

        return all_content_flags, all_tokens

    def __call__(self, pipeline_args: Dict[str, Any]) -> torch.Tensor:
        """
        Process encoder hidden states based on importance rankings and keep ratios.

        This is the main processing method that combines importance calculation,
        token selection, and state interpolation to produce the final processed
        encoder hidden states.

        Args:
            pipeline_args (Dict[str, Any]): Pipeline arguments containing:
                - encoder_hidden_states: Input encoder hidden states tensor
                - denoising_step: Current denoising step (optional)
                - index_block: Current transformer block index (optional)

        Returns:
            torch.Tensor: Processed encoder hidden states with shape [2, seq_len, hidden_dim]
                         where the first element is the processed result and the second
                         element is the original positive batch.
        """
        # Update current processing information
        self.current_step = pipeline_args.get("denoising_step", None)
        self.current_block = pipeline_args.get("index_block", None)

        # Synchronize step information to importance calculator
        if hasattr(self.importance_calculator, "current_step"):
            self.importance_calculator.current_step = self.current_step
            self.importance_calculator.current_block = self.current_block

        # Extract processing parameters
        keep_ratio = self.process_params["keep_ratio"]
        separate_clip_t5 = self.process_params["separate_clip_t5"]
        positive_encoder_hidden_states = pipeline_args["encoder_hidden_states"][1:2]
        negative_encoder_hidden_states = self.empty_encoder_hidden_states

        # Ensure dtype consistency between negative and positive states
        if negative_encoder_hidden_states.dtype != positive_encoder_hidden_states.dtype:
            negative_encoder_hidden_states = negative_encoder_hidden_states.to(
                positive_encoder_hidden_states.dtype
            )
        # Determine if attention needs to be calculated; no calculation if ratio is 1 or 0
        if self.debug:
            jump_calculate_attention = False
        else:
            if keep_ratio in [
                {"all": 1},
                {"all": 0},
                {"content": 1, "padding": 0},
                {"content": 1, "padding": 1},
                {"content": 0, "padding": 1},
                {"content": 0, "padding": 0},
            ]:
                jump_calculate_attention = True
            else:
                jump_calculate_attention = False

        scores = None
        # Check for cached importance scores
        if hasattr(self, 'cached_sorted_indices') and self.cached_sorted_indices is not None:
            sorted_indices = self.cached_sorted_indices
            scores = self.cached_final_scores
        elif jump_calculate_attention:
            sorted_indices = [i for i in range(len(self.all_content_flags))]
        elif (
            self.process_params.get("all_use_first_step_importance")
            and self.first_step_importance is None
        ):
            if self.process_params.get("use_random_sorted_indices"):
                sorted_indices = [i for i in range(len(self.all_content_flags))]
                # Use independent random generator if random_index_seed is set
                if self._index_random is not None:
                    self._index_random.shuffle(sorted_indices)
                else:
                    random.shuffle(sorted_indices)
            else:
                sorted_indices, scores = self.importance_calculator(pipeline_args)
            self.first_step_importance = sorted_indices
        elif (
            self.process_params.get("all_use_first_step_importance")
            and self.first_step_importance is not None
        ):
            sorted_indices = self.first_step_importance
        elif self.process_params.get("use_random_sorted_indices"):
            sorted_indices = [i for i in range(len(self.all_content_flags))]
            # Use independent random generator if random_index_seed is set
            if self._index_random is not None:
                self._index_random.shuffle(sorted_indices)
            else:
                random.shuffle(sorted_indices)
        else:
            # Calculate token importance rankings
            sorted_indices, scores = self.importance_calculator(pipeline_args)

        # Generate keep mask based on ratios and token content status
        keep_mask, kept_indices = self.get_keep_mask(
            sorted_indices, separate_clip_t5, ratio=keep_ratio
        )

        # Record importance data for debugging if enabled
        if self.debug and scores is not None:
            # Check recording strategy based on all_use_first_step_importance
            should_record = False
            if self.process_params.get("all_use_first_step_importance", False):
                # Record only if this is the first record
                should_record = len(self.debug_data["importance_records"]) == 0
            else:
                # Record all steps
                should_record = True

            if should_record:
                importance_record = {
                    "denoising_step": self.current_step,
                    "transformer_block": self.current_block,
                    "scores": (
                        scores.cpu().tolist() if torch.is_tensor(scores) else scores
                    ),
                    "sorted_indices": (
                        sorted_indices
                        if isinstance(sorted_indices, list)
                        else sorted_indices.cpu().tolist()
                    ),
                    "kept_indices": kept_indices,
                }
                self.debug_data["importance_records"].append(importance_record)

        # Ensure keep_mask has consistent dtype with encoder hidden states
        if keep_mask.dtype != positive_encoder_hidden_states.dtype:
            keep_mask = keep_mask.to(positive_encoder_hidden_states.dtype)

        # Expand keep_mask dimensions to match encoder_hidden_states shape
        # [seq_length] -> [1, seq_length, 1]
        keep_mask = keep_mask.unsqueeze(0).unsqueeze(-1)

        # Interpolate between positive and negative states using the mask
        result = (
            keep_mask * positive_encoder_hidden_states
            + (1 - keep_mask) * negative_encoder_hidden_states
        )

        # Concatenate to maintain batch=2 format: [result, positive_batch]
        process_result = torch.cat([result, positive_encoder_hidden_states], dim=0)
        return process_result

    def save_debug_data(self, output_root: str = "outputs/debug_output"):
        """
        Save all collected debug data to structured JSON files and generate visualizations.

        This method creates the following output structure:
        - outputs/debug_output/metadata.json: Static information (timestamp, prompt, etc.)
        - outputs/debug_output/importance_data.json: Dynamic importance calculations
        - outputs/debug_output/figures/: Directory containing score visualizations

        Args:
            output_root (str): Root directory for debug outputs
        """
        if not self.debug or not hasattr(self, "debug_data"):
            print("Debug mode not enabled or no debug data available.")
            return

        # Create directory structure
        figures_dir = os.path.join(output_root, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Save metadata.json
        metadata_path = os.path.join(output_root, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.debug_data["metadata"], f, indent=2, ensure_ascii=False)
        print(f"Metadata saved to: {metadata_path}")

        # Save importance_data.json
        importance_path = os.path.join(output_root, "importance_data.json")
        with open(importance_path, "w", encoding="utf-8") as f:
            json.dump(self.debug_data["importance_records"], f, indent=2)
        print(f"Importance data saved to: {importance_path}")

        # Generate figures based on importance records
        self._generate_figures(self.debug_data["importance_records"], figures_dir)

        print(f"Debug data processing complete. Output saved to: {output_root}")

    def _generate_figures(self, importance_records: List[Dict], figures_dir: str):
        """
        Generate score visualization figures from importance records.

        Args:
            importance_records (List[Dict]): List of importance calculation records
            figures_dir (str): Directory to save generated figures
        """
        if not importance_records:
            print("No importance records available for visualization.")
            return

        # Get token content flags from metadata
        token_content_flags = self.debug_data["metadata"].get("token_content_flags", [])

        for i, record in enumerate(importance_records):
            try:
                scores = record.get("scores", [])
                kept_indices = record.get("kept_indices", [])
                if not scores:
                    continue

                # Convert to numpy array for plotting
                if isinstance(scores, list):
                    scores_array = np.array(scores)
                else:
                    scores_array = scores

                # Create figure
                plt.figure(figsize=(20, 8))

                # Generate token indices for x-axis
                token_indices = np.arange(len(scores_array))

                # Create color array: green for kept tokens, blue for discarded tokens
                colors = ['#2ecc71' if idx in kept_indices else '#3498db' 
                         for idx in token_indices]

                # Create bar plot with different colors
                plt.bar(token_indices, scores_array, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

                # Find the boundary between content and padding tokens
                if token_content_flags and len(token_content_flags) == len(scores_array):
                    # Find first False (padding token)
                    content_end = None
                    for idx, is_content in enumerate(token_content_flags):
                        if not is_content:
                            content_end = idx - 0.5  # Place line between tokens
                            break
                    
                    # Draw vertical line to separate content and padding
                    if content_end is not None:
                        plt.axvline(x=content_end, color='red', linestyle='--', linewidth=2, 
                                   label='Content/Padding Boundary')

                # Add title and labels
                step = record.get("denoising_step", "unknown")
                block = record.get("transformer_block", "unknown")
                plt.title(f"Step {step} Block {block} - Token Importance Scores\n"
                         f"Green: Kept tokens ({len(kept_indices)}), Blue: Discarded tokens", 
                         fontsize=14, fontweight='bold')
                plt.xlabel("Token Index", fontsize=12)
                plt.ylabel("Importance Score", fontsize=12)
                plt.grid(True, linestyle="--", alpha=0.3, axis='y')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#2ecc71', edgecolor='black', label=f'Kept tokens ({len(kept_indices)})'),
                    Patch(facecolor='#3498db', edgecolor='black', label=f'Discarded tokens ({len(scores_array) - len(kept_indices)})')
                ]
                if content_end is not None:
                    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Content/Padding Boundary'))
                plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
                
                plt.tight_layout()

                # Save figure
                figure_name = f"step_{step:02d}_block_{block:02d}_scores.png"
                figure_path = os.path.join(figures_dir, figure_name)
                plt.savefig(figure_path, dpi=150, bbox_inches="tight")
                plt.close()

                print(f"Figure saved: {figure_path}")

            except Exception as e:
                print(f"Error generating figure for record {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
