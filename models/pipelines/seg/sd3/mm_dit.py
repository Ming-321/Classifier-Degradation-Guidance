"""
SEG (Self-attention Guidance) implementation for Stable Diffusion 3.

This module implements SEG by perturbing image self-attention queries through spatial blurring.
Supports both CFG+SEG and SEG-only modes.
"""

from typing import Optional
import torch
import math
import torch.nn.functional as F


def gaussian_blur_2d(img, kernel_size, sigma):
    """
    Apply 2D Gaussian blur to an image tensor.
    
    Args:
        img: Image tensor of shape (B, C, H, W)
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian distribution
    
    Returns:
        Blurred image tensor of the same shape
    """
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class SEGJointAttnProcessor:
    """
    Attention processor for SEG (Self-attention Guidance) for SD3's JointAttention.
    
    This processor perturbs the image query vectors through spatial blurring before joint 
    attention computation. The perturbation degrades spatial self-attention, allowing SEG
    to guide the generation towards better quality by comparing perturbed and original outputs.
    
    Args:
        blur_sigma (float): Standard deviation for Gaussian blur. Values > inf_blur_threshold 
            result in uniform blur (all queries become mean value).
        do_cfg (bool): Whether CFG is enabled. Affects batch dimension:
            - True: batch = [uncond, cond, cond_perturbed] (size 3)
            - False: batch = [original, perturbed] (size 2)
        inf_blur_threshold (float): Threshold above which uniform blur is used instead of Gaussian.
    """
    
    def __init__(self, blur_sigma=1.0, do_cfg=True, inf_blur_threshold=9999.0):
        self.blur_sigma = blur_sigma
        self.do_cfg = do_cfg
        if self.blur_sigma > inf_blur_threshold:
            self.inf_blur = True
        else:
            self.inf_blur = False

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
        Apply SEG-enabled joint attention.
        
        The core SEG mechanism: perturb image queries by spatial blurring before attention 
        computation. This creates a degraded branch that guidance can compare against.
        
        Steps:
        1. Compute Q, K, V for image (hidden_states)
        2. Apply normalization if present
        3. Perturb queries for the perturbed branch by spatial blurring
        4. Compute Q, K, V for text (encoder_hidden_states)  
        5. Concatenate image and text Q, K, V
        6. Compute joint attention
        7. Split and project outputs
        """
        residual = hidden_states
        
        batch_size = hidden_states.shape[0]
        
        # Step 1: Compute image projections (Q, K, V from hidden_states)
        # This is the standard attention projection for the image latent
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        # Reshape to multi-head format: (batch, seq_len, dim) -> (batch, heads, seq_len, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Apply query/key normalization (SD3 uses RMSNorm on queries and keys)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Step 2: Perturb image queries for SEG guidance
        # CRITICAL: This is the core SEG perturbation
        
        # Calculate spatial dimensions from sequence length (assumes square latents)
        img_seq_len = hidden_states.shape[1]
        height = width = int(math.sqrt(img_seq_len))
        
        if self.do_cfg:
            # CFG + SEG mode: batch contains [uncond, cond, cond_perturbed]
            # Split query into three branches
            query_uncond, query_org, query_ptb = query.chunk(3)
            
            # Reshape perturbed query to spatial dimensions for blurring
            # Transform: (batch//3, heads, seq_len, head_dim) -> (batch//3, heads*head_dim, H, W)
            query_ptb = query_ptb.permute(0, 1, 3, 2).reshape(
                batch_size // 3, attn.heads * head_dim, height, width
            )
            
            # Apply perturbation (blur the spatial structure)
            if not self.inf_blur:
                # Gaussian blur with computed kernel size
                kernel_size = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
                query_ptb = gaussian_blur_2d(query_ptb, kernel_size, self.blur_sigma)
            else:
                # Uniform blur: replace all spatial positions with mean
                query_ptb[:] = query_ptb.mean(dim=(-2, -1), keepdim=True)
            
            # Reshape back to sequence format
            # Transform: (batch//3, heads*head_dim, H, W) -> (batch//3, heads, seq_len, head_dim)
            query_ptb = query_ptb.reshape(
                batch_size // 3, attn.heads, head_dim, height * width
            ).permute(0, 1, 3, 2)
            
            # Concatenate all three branches back
            query = torch.cat((query_uncond, query_org, query_ptb), dim=0)
        else:
            # SEG only mode: batch contains [original, perturbed]
            # Split query into two branches
            query_org, query_ptb = query.chunk(2)
            
            # Reshape perturbed query to spatial dimensions
            query_ptb = query_ptb.permute(0, 1, 3, 2).reshape(
                batch_size // 2, attn.heads * head_dim, height, width
            )
            
            # Apply perturbation
            if not self.inf_blur:
                kernel_size = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
                query_ptb = gaussian_blur_2d(query_ptb, kernel_size, self.blur_sigma)
            else:
                query_ptb[:] = query_ptb.mean(dim=(-2, -1), keepdim=True)
            
            # Reshape back to sequence format
            query_ptb = query_ptb.reshape(
                batch_size // 2, attn.heads, head_dim, height * width
            ).permute(0, 1, 3, 2)
            
            # Concatenate both branches back
            query = torch.cat((query_org, query_ptb), dim=0)
        
        # Step 3: Process text/context Q, K, V (encoder_hidden_states)
        # In SD3's JointAttention, we concatenate image and text Q, K, V before attention.
        if encoder_hidden_states is not None:
            # Project text embeddings to Q, K, V using separate projections
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            
            # Reshape text projections to multi-head format
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            
            # Apply normalization to text queries and keys
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            # Step 4: Concatenate image and text Q, K, V for joint attention
            # Image queries are already perturbed (if in perturbed branch)
            # Text queries remain unchanged - only image queries are perturbed by SEG
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
        
        # Step 5: Compute scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # Step 6: Split outputs back into image and text (SD3-specific)
        if encoder_hidden_states is not None:
            # Separate the concatenated output back into image and text parts
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            # Project text output if not in context_pre_only mode
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        # Step 7: Final output projection
        hidden_states = attn.to_out[0](hidden_states)  # Linear projection
        hidden_states = attn.to_out[1](hidden_states)  # Dropout
        
        # Return image and text outputs (SD3's JointAttention returns both)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
