# Copyright 2025 The HuggingFace Team and Classifier Degradation Guidance Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import numpy as np

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import (
    PipelineImageInput,
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    replace_example_docstring,
)

from ..pag_utils import PAGMixin
from .pag_attn_processor import PAGFluxAttnProcessor, PAGCFGFluxAttnProcessor

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from models.pipelines.pag.flux.pipeline import PAGFluxPipeline

        >>> pipe = PAGFluxPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.1-schnell", 
        ...     torch_dtype=torch.bfloat16,
        ...     pag_applied_layers=["transformer_blocks.10"]
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt, pag_scale=3.0, guidance_scale=3.5).images[0]
        >>> image.save("pag_flux.png")
        ```
"""


class PAGFluxPipeline(FluxPipeline, PAGMixin):
    """
    PAG (Perturbed Attention Guidance) enabled Flux Pipeline for text-to-image generation.
    
    This pipeline extends the base FluxPipeline to support PAG through dynamic attention 
    processor replacement. PAG perturbs self-attention between image patches to provide 
    additional guidance during generation.
    
    Key features:
    - Two guidance modes: PAG only, CFG+PAG combined
    - Configurable identity mask perturbation
    - Layer-specific PAG application
    - Compatible with standard Flux model weights
    
    Args:
        Same as FluxPipeline, with additional PAG parameters available in the __call__ method
        (pag_scale, pag_applied_layers, pag_adaptive_scale)
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["feature_extractor", "image_encoder"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        pag_applied_layers: Union[str, List[str]] = "transformer_blocks.10",
        feature_extractor=None,
        image_encoder=None,
    ):
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
        )
        
        # Set up PAG attention processors
        self.set_pag_applied_layers(
            pag_applied_layers,
            pag_attn_processors=(PAGCFGFluxAttnProcessor(), PAGFluxAttnProcessor()),
        )

    @property
    def do_classifier_free_guidance(self):
        """Check if classifier-free guidance is enabled based on true_cfg_scale."""
        return self._true_cfg_scale > 1.0

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # PAG specific parameters
        pag_scale: float = 3.0,
        pag_adaptive_scale: float = 0.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation with PAG.

        Examples:

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and `text_encoder_2`.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*):
                The height in pixels of the generated image.
            width (`int`, *optional*):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 28):
                The number of denoising steps.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in Classifier-Free Diffusion Guidance.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A generator to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings.
            ip_adapter_image (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter.
            negative_ip_adapter_image (`PipelineImageInput`, *optional*):
                Optional negative image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated negative image embeddings for IP-Adapter.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a FluxPipelineOutput instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary passed along to the AttentionProcessor.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            max_sequence_length (`int`, defaults to 512):
                Maximum sequence length to use with the prompt.
            pag_scale (`float`, *optional*, defaults to 3.0):
                The scale factor for PAG (Perturbed Attention Guidance). Set to 0 to disable PAG.
                Higher values increase the guidance strength. Typical range: 0.5-5.0.
            pag_adaptive_scale (`float`, *optional*, defaults to 0.0):
                Adaptive scale factor for PAG that varies with timestep. When > 0, PAG scale 
                decreases as denoising progresses.

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`:
            FluxPipelineOutput if `return_dict` is True, otherwise a tuple.
        """
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._true_cfg_scale = true_cfg_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        
        # PAG parameters
        self._pag_scale = pag_scale
        self._pag_adaptive_scale = pag_adaptive_scale

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        
        # Check if we have negative prompt for CFG
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        
        # 3. Encode prompts
        (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        
        if do_true_cfg:
            (negative_prompt_embeds, negative_pooled_prompt_embeds, _) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # Prepare embeddings for PAG
        # PAG requires duplicating the conditional embeddings for original and perturbed paths
        if self.do_perturbed_attention_guidance:
            prompt_embeds = self._prepare_perturbed_attention_guidance(
                prompt_embeds, negative_prompt_embeds if do_true_cfg else None, do_true_cfg
            )
            pooled_prompt_embeds = self._prepare_perturbed_attention_guidance(
                pooled_prompt_embeds, negative_pooled_prompt_embeds if do_true_cfg else None, do_true_cfg
            )
            
            # Also need to duplicate text_ids for PAG
            if do_true_cfg:
                text_ids = torch.cat([text_ids] * 3, dim=0)
            else:
                text_ids = torch.cat([text_ids] * 2, dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
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
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Handle guidance embedding
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Prepare IP Adapter image embeddings if needed
        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}
            
        # Handle IP adapter (simplified)
        image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 7. Set PAG attention processors
        if self.do_perturbed_attention_guidance:
            self._set_pag_attn_processor(self.pag_applied_layers, do_true_cfg)

        # 8. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Expand latents for PAG
                if self.do_perturbed_attention_guidance:
                    if do_true_cfg:
                        # Three branches: uncond, cond_original, cond_perturbed
                        latent_model_input = torch.cat([latents] * 3)
                    else:
                        # Two branches: original, perturbed
                        latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # Broadcast timestep
                timestep = t.expand(latent_model_input.shape[0])

                # Prepare image embeds for joint attention
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                # Transformer forward pass
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Apply PAG guidance
                if self.do_perturbed_attention_guidance:
                    noise_pred = self._apply_perturbed_attention_guidance(
                        noise_pred, do_true_cfg, guidance_scale, t
                    )

                # Compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 9. Restore original attention processors
        if self.do_perturbed_attention_guidance:
            from diffusers.models.attention_processor import FluxAttnProcessor2_0
            for layer_id in self.pag_applied_layers:
                for name, module in self.transformer.named_modules():
                    if layer_id in name and hasattr(module, 'processor'):
                        module.processor = FluxAttnProcessor2_0()

        # 10. Decode latents
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

