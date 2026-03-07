"""
SFG (Segmentation-free Guidance) Pipeline for Stable Diffusion 3.

Reference: "Segmentation-free guidance for text-to-image diffusion models" (CVPR 2024 Workshop)

Key mechanism:
- Two-phase denoising:
  1. Phase 1 (steps 0 to sfg_start_step-1): Standard CFG
  2. Phase 2 (steps sfg_start_step to end): SFG
  
- SFG formula: ε̃ = (1 + w̄)ε(z,c) - w̄ε̄(z,c)
  - Both branches use conditional input c (NOT unconditional!)
  - ε̄ has modified attention: max text token weight * (-a)
"""

from typing import Any, Callable, Dict, List, Optional, Union
import torch

from diffusers import StableDiffusion3Pipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
    calculate_shift,
)
from diffusers.utils import replace_example_docstring
from diffusers.models.attention_processor import JointAttnProcessor2_0
from .mm_dit import SFGJointAttnProcessor

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from models.pipelines.sfg.sd3.pipeline import StableDiffusion3SFGPipeline

        >>> pipe = StableDiffusion3SFGPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3_sfg.png")
        ```
"""


class StableDiffusion3SFGPipeline(StableDiffusion3Pipeline):
    """
    SFG (Segmentation-free Guidance) enabled Stable Diffusion 3 Pipeline.

    This pipeline extends the base StableDiffusion3Pipeline to support SFG through
    two-phase denoising:
    - Phase 1: Standard CFG (conditional vs unconditional)
    - Phase 2: SFG (conditional vs attention-modified conditional)

    Key features:
    - Two-phase guidance switching based on sfg_start_ratio
    - Dynamic attention processor replacement for SFG phase
    - Per-patch attention weight modification
    
    Args:
        Same as StableDiffusion3Pipeline, with additional SFG parameters in __call__
    """

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        sfg_guidance_scale: float = 2.0,
        sfg_scale: float = 10.0,
        sfg_start_ratio: float = 0.5,
        sfg_applied_layers_index: List[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        mu: Optional[float] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]}`, *optional*):
                The prompt or prompts to guide the image generation.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                CFG guidance scale for Phase 1 (w in paper).
            sfg_guidance_scale (`float`, *optional*, defaults to 2.0):
                SFG guidance scale for Phase 2 (w̄ in paper).
            sfg_scale (`float`, *optional*, defaults to 10.0):
                Attention modification scale (a in paper). Max attention weight * (-a).
            sfg_start_ratio (`float`, *optional*, defaults to 0.5):
                Ratio of total steps at which to switch from CFG to SFG.
                sfg_start_step = int(num_inference_steps * sfg_start_ratio)
            sfg_applied_layers_index (`List[int]`, *optional*):
                List of transformer block indices to apply SFG. None means all layers.
            num_inference_steps (`int`, *optional*, defaults to 28):
                Total number of denoising steps.
            ... (other standard SD3 parameters)

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        
        # SFG parameters
        self._sfg_guidance_scale = sfg_guidance_scale
        self._sfg_scale = sfg_scale
        self._sfg_start_ratio = sfg_start_ratio
        # Default: apply SFG to all layers
        self._sfg_applied_layers_index = sfg_applied_layers_index if sfg_applied_layers_index is not None else list(range(len(self.transformer.transformer_blocks)))

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        
        # 3. Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # Store original embeddings for phase switching
        original_prompt_embeds = prompt_embeds.clone()
        original_pooled_prompt_embeds = pooled_prompt_embeds.clone()
        
        # For CFG phase: [negative, positive]
        cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        cfg_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        
        # For SFG phase: [positive, positive] (both use conditional!)
        sfg_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
        sfg_pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height_latent, width_latent = latents.shape
            image_seq_len = (height_latent // self.transformer.config.patch_size) * (
                width_latent // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
            
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        # Calculate SFG start step
        sfg_start_step = int(num_inference_steps * self._sfg_start_ratio)

        # 6. Prepare image embeddings (IP-Adapter)
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # Store original attention processors
        original_attn_processors = {}
        for layer_idx in self._sfg_applied_layers_index:
            if layer_idx < len(self.transformer.transformer_blocks):
                original_attn_processors[layer_idx] = self.transformer.transformer_blocks[layer_idx].attn.processor

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Determine which phase we're in
                is_sfg_phase = i >= sfg_start_step
                
                if not is_sfg_phase:
                    # Phase 1: Standard CFG
                    # prompt_embeds = [negative, positive]
                    current_prompt_embeds = cfg_prompt_embeds
                    current_pooled_prompt_embeds = cfg_pooled_prompt_embeds
                    latent_model_input = torch.cat([latents] * 2)
                    
                    # Restore original attention processors
                    for layer_idx in self._sfg_applied_layers_index:
                        if layer_idx < len(self.transformer.transformer_blocks):
                            self.transformer.transformer_blocks[layer_idx].attn.processor = JointAttnProcessor2_0()
                else:
                    # Phase 2: SFG
                    # prompt_embeds = [positive, positive] - BOTH conditional!
                    current_prompt_embeds = sfg_prompt_embeds
                    current_pooled_prompt_embeds = sfg_pooled_prompt_embeds
                    latent_model_input = torch.cat([latents] * 2)
                    
                    # Replace attention processors with SFG processor
                    sfg_processor = SFGJointAttnProcessor(
                        sfg_scale=self._sfg_scale,
                        exclude_bos=True
                    )
                    for layer_idx in self._sfg_applied_layers_index:
                        if layer_idx < len(self.transformer.transformer_blocks):
                            self.transformer.transformer_blocks[layer_idx].attn.processor = sfg_processor

                # Broadcast timestep
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=current_prompt_embeds,
                    pooled_projections=current_pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Apply guidance
                if not is_sfg_phase:
                    # Phase 1: CFG
                    # ε̃ = ε_uncond + w * (ε_cond - ε_uncond)
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # Phase 2: SFG
                    # ε̃ = (1 + w̄) * ε_cond - w̄ * ε̄_cond
                    # which is equivalent to: ε̃ = ε_cond + w̄ * (ε_cond - ε̄_cond)
                    noise_pred_cond, noise_pred_perturbed = noise_pred.chunk(2)
                    noise_pred = noise_pred_cond + self._sfg_guidance_scale * (noise_pred_cond - noise_pred_perturbed)

                # Compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # Restore original attention processors
        for layer_idx, processor in original_attn_processors.items():
            if layer_idx < len(self.transformer.transformer_blocks):
                self.transformer.transformer_blocks[layer_idx].attn.processor = processor

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
