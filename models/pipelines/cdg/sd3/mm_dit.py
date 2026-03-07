from diffusers.models.transformers import SD3Transformer2DModel
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class CDGSD3Transformer2DModel(SD3Transformer2DModel):
    """
    Enhanced SD3 Transformer 2D Model with Context-aware Weight Generation (CDG) support.

    This model extends the base SD3Transformer2DModel to support token processing
    and selective weight generation for improved text-to-image generation quality.

    Inherits from SD3Transformer2DModel and adds support for:
    - Token importance calculation and processing
    - Context-aware weight adjustment during diffusion
    - Layer skipping capabilities for guidance control
    - Debug and monitoring capabilities
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        token_processor: Optional[Callable] = None,
        process_params: Optional[Dict[str, Any]] = None,
        denoising_step: Optional[int] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.
            token_processor (`Callable`, *optional*):
                Token processor for CDG processing.
            process_params (`dict`, *optional*):
                Parameters for CDG processing.
            denoising_step (`int`, *optional*):
                Current denoising step for debug recording.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(
            hidden_states
        )  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if (
            joint_attention_kwargs is not None
            and "ip_adapter_image_embeds" in joint_attention_kwargs
        ):
            ip_adapter_image_embeds = joint_attention_kwargs.pop(
                "ip_adapter_image_embeds"
            )
            ip_hidden_states, ip_temb = self.image_proj(
                ip_adapter_image_embeds, timestep
            )

            joint_attention_kwargs.update(
                ip_hidden_states=ip_hidden_states, temb=ip_temb
            )

        for index_block, block in enumerate(self.transformer_blocks):
            # Skip specified layers
            is_skip = (
                True
                if skip_layers is not None and index_block in skip_layers
                else False
            )

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        joint_attention_kwargs,
                    )
                )
            elif not is_skip:
                # CDG processing
                if encoder_hidden_states.shape[0] == 3:
                    encoder_hidden_states_ = encoder_hidden_states[0:2]
                    if index_block == process_params["process_index"]:
                        pipeline_args = {
                            "pipeline_name": "sd3",
                            "index_block": index_block,
                            "hidden_states": hidden_states,
                            "encoder_hidden_states": encoder_hidden_states_,
                            "temb": temb,
                            "denoising_step": denoising_step,
                        }
                        encoder_hidden_states[0:2] = token_processor(pipeline_args)
                else:
                    if index_block == process_params["process_index"]:
                        pipeline_args = {
                            "pipeline_name": "sd3",
                            "index_block": index_block,
                            "hidden_states": hidden_states,
                            "encoder_hidden_states": encoder_hidden_states,
                            "temb": temb,
                            "denoising_step": denoising_step,
                        }
                        encoder_hidden_states = token_processor(pipeline_args)
                # ------
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if (
                block_controlnet_hidden_states is not None
                and block.context_pre_only is False
            ):
                interval_control = len(self.transformer_blocks) / len(
                    block_controlnet_hidden_states
                )
                hidden_states = (
                    hidden_states
                    + block_controlnet_hidden_states[
                        int(index_block / interval_control)
                    ]
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
