from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass

'''
NO TRAINING, NO PROBLEM: RETHINKING  CLASSIFIER-FREE GUIDANCE FOR DIFFUSION MODELS
(https://arxiv.org/abs/2407.02687)
'''



@dataclass
class EncoderConfig:
    """Configuration for a single encoder"""

    name: str
    tokenizer: Any
    text_encoder: Any
    seq_length: int
    vocab_size: int
    special_tokens: Dict[str, int]
    embedding_dim: int


@dataclass
class ModelConfig:
    """Complete configuration for the model"""

    model_name: str
    encoders: List[EncoderConfig]

    def get_encoder(self, name: str) -> Optional[EncoderConfig]:
        """Get encoder configuration by name"""
        for encoder in self.encoders:
            if encoder.name == name:
                return encoder
        return None


class RandomTokenGenerator(ABC):
    """Abstract base class for random token generators"""

    def __init__(self, config: ModelConfig, device: str = "cuda"):
        self.config = config
        self.device = device

    @abstractmethod
    def generate_random_embeddings(
        self,
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        return_texts: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...], List[str]]]:
        """Generates random embeddings"""
        pass

    def _generate_random_indices(
        self,
        vocab_size: int,
        seq_length: int,
        generator: Optional[torch.Generator] = None,
        avoid_special_tokens: bool = True,
    ) -> torch.Tensor:
        """Generates random token indices"""
        if avoid_special_tokens:
            # Avoid special tokens, usually at the beginning and end of the vocab
            low = max(1, int(vocab_size * 0.001))  # Avoid first 0.1%
            high = min(vocab_size - 100, int(vocab_size * 0.95))  # Avoid last 5%
        else:
            low, high = 1, vocab_size - 1

        random_idx = torch.randint(
            low=low,
            high=high,
            size=(1, seq_length),
            device=self.device,
            generator=generator,
        )

        return random_idx

    def _add_special_tokens(
        self, token_ids: torch.Tensor, special_tokens: Dict[str, int]
    ):
        """Adds special tokens (e.g., beginning, end tokens)"""
        if (
            "bos_token_id" in special_tokens
            and special_tokens["bos_token_id"] is not None
        ):
            token_ids[0, 0] = special_tokens["bos_token_id"]
        if (
            "eos_token_id" in special_tokens
            and special_tokens["eos_token_id"] is not None
        ):
            token_ids[0, -1] = special_tokens["eos_token_id"]
        return token_ids

    def _decode_tokens(self, token_ids: torch.Tensor, tokenizer: Any) -> str:
        """Decodes token indices into text"""
        try:
            return tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)
        except Exception:
            return "Failed to decode"


class SD3RandomTokenGenerator(RandomTokenGenerator):
    """Random token generator for SD3 model"""

    def generate_random_embeddings(
        self,
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        return_texts: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor], List[str]],
    ]:
        """
        Generates random embeddings for SD3
        Returns: (prompt_embeds, pooled_prompt_embeds) or ((prompt_embeds, pooled_prompt_embeds), decoded_texts)
        """

        # Get encoder configurations
        clip_l = self.config.get_encoder("clip_l")
        clip_g = self.config.get_encoder("clip_g")
        t5 = self.config.get_encoder("t5")

        decoded_texts = []

        # Generate CLIP-L embeddings
        clip_l_embeds, clip_l_pooled, clip_l_text = self._generate_clip_embeddings(
            clip_l, generator, return_texts
        )

        # Generate CLIP-G embeddings
        clip_g_embeds, clip_g_pooled, clip_g_text = self._generate_clip_embeddings(
            clip_g, generator, return_texts
        )

        # Generate T5 embeddings
        t5_embeds, t5_text = self._generate_t5_embeddings(t5, generator, return_texts)

        # Concatenate CLIP embeddings
        clip_combined_embeds = torch.cat([clip_l_embeds, clip_g_embeds], dim=-1)
        clip_combined_pooled = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)

        # Align dimensions
        clip_combined_embeds = torch.nn.functional.pad(
            clip_combined_embeds,
            (0, t5_embeds.shape[-1] - clip_combined_embeds.shape[-1]),
        )

        # Final concatenation
        final_embeds = torch.cat([clip_combined_embeds, t5_embeds], dim=-2)

        # Handle batch dimension
        final_embeds = final_embeds.repeat(batch_size * num_images_per_prompt, 1, 1)
        final_pooled = clip_combined_pooled.repeat(
            batch_size * num_images_per_prompt, 1
        )

        if return_texts:
            decoded_texts = [
                f"CLIP-L: {clip_l_text}",
                f"CLIP-G: {clip_g_text}",
                f"T5: {t5_text}",
            ]
            return (final_embeds, final_pooled), decoded_texts

        return final_embeds, final_pooled

    def _generate_clip_embeddings(
        self,
        encoder_config: EncoderConfig,
        generator: Optional[torch.Generator],
        return_texts: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[str]]:
        """Generates CLIP encoder embeddings"""

        # Generate random indices
        random_idx = self._generate_random_indices(
            encoder_config.vocab_size, encoder_config.seq_length, generator
        )

        # Add special tokens
        random_idx = self._add_special_tokens(random_idx, encoder_config.special_tokens)

        # Generate embeddings
        with torch.no_grad():
            outputs = encoder_config.text_encoder(
                random_idx.to(self.device), output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-2]  # Second to last layer
            pooled_output = outputs[0]

        decoded_text = None
        if return_texts:
            decoded_text = self._decode_tokens(random_idx, encoder_config.tokenizer)

        return hidden_states, pooled_output, decoded_text

    def _generate_t5_embeddings(
        self,
        encoder_config: EncoderConfig,
        generator: Optional[torch.Generator],
        return_texts: bool = False,
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """Generates T5 encoder embeddings"""

        if encoder_config.text_encoder is None:
            # If no T5 encoder, return zero tensor
            zero_embeds = torch.zeros(
                (1, encoder_config.seq_length, encoder_config.embedding_dim),
                device=self.device,
                dtype=torch.float16,
            )
            return zero_embeds, "No T5 encoder"

        # Generate random indices
        random_idx = self._generate_random_indices(
            encoder_config.vocab_size, encoder_config.seq_length, generator
        )

        # Generate embeddings
        with torch.no_grad():
            embeddings = encoder_config.text_encoder(random_idx.to(self.device))[0]

        decoded_text = None
        if return_texts:
            decoded_text = self._decode_tokens(random_idx, encoder_config.tokenizer)

        return embeddings, decoded_text


class FluxRandomTokenGenerator(RandomTokenGenerator):
    """Random token generator for Flux model"""

    def generate_random_embeddings(
        self,
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        return_texts: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor], List[str]],
    ]:
        """
        Generates random embeddings for Flux
        Returns: (prompt_embeds, pooled_prompt_embeds) or ((prompt_embeds, pooled_prompt_embeds), decoded_texts)
        """

        # Get encoder configurations
        clip_l = self.config.get_encoder("clip_l")
        t5 = self.config.get_encoder("t5")

        decoded_texts = []

        # Generate CLIP-L embeddings (only for pooled_prompt_embeds)
        clip_pooled_embeds, clip_text = self._generate_clip_pooled_embeddings(
            clip_l, generator, return_texts
        )

        # Generate T5 embeddings (for prompt_embeds)
        t5_embeds, t5_text = self._generate_t5_embeddings(t5, generator, return_texts)

        # Handle batch dimension
        final_prompt_embeds = t5_embeds.repeat(batch_size * num_images_per_prompt, 1, 1)
        final_pooled_embeds = clip_pooled_embeds.repeat(
            batch_size * num_images_per_prompt, 1
        )

        if return_texts:
            decoded_texts = [f"CLIP-L (pooled): {clip_text}", f"T5 (prompt): {t5_text}"]
            return (final_prompt_embeds, final_pooled_embeds), decoded_texts

        return final_prompt_embeds, final_pooled_embeds

    def _generate_clip_pooled_embeddings(
        self,
        encoder_config: EncoderConfig,
        generator: Optional[torch.Generator],
        return_texts: bool = False,
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """Generates pooled embeddings for CLIP encoder"""

        # Generate random indices
        random_idx = self._generate_random_indices(
            encoder_config.vocab_size, encoder_config.seq_length, generator
        )

        # Add special tokens
        random_idx = self._add_special_tokens(random_idx, encoder_config.special_tokens)

        # Generate embeddings
        with torch.no_grad():
            outputs = encoder_config.text_encoder(
                random_idx.to(self.device), output_hidden_states=False
            )
            # Use pooled output
            pooled_output = outputs.pooler_output

        decoded_text = None
        if return_texts:
            decoded_text = self._decode_tokens(random_idx, encoder_config.tokenizer)

        return pooled_output, decoded_text

    def _generate_t5_embeddings(
        self,
        encoder_config: EncoderConfig,
        generator: Optional[torch.Generator],
        return_texts: bool = False,
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """Generates T5 encoder embeddings"""

        if encoder_config.text_encoder is None:
            # If no T5 encoder, return zero tensor
            zero_embeds = torch.zeros(
                (1, encoder_config.seq_length, encoder_config.embedding_dim),
                device=self.device,
                dtype=torch.float16,
            )
            return zero_embeds, "No T5 encoder"

        # Generate random indices
        random_idx = self._generate_random_indices(
            encoder_config.vocab_size, encoder_config.seq_length, generator
        )

        # Generate embeddings
        with torch.no_grad():
            embeddings = encoder_config.text_encoder(
                random_idx.to(self.device), output_hidden_states=False
            )[0]

        decoded_text = None
        if return_texts:
            decoded_text = self._decode_tokens(random_idx, encoder_config.tokenizer)

        return embeddings, decoded_text




def create_model_config(pipeline, model_name: str) -> ModelConfig:
    """Creates model configuration based on the pipeline"""

    if model_name.lower() == "sd3":
        return ModelConfig(
            model_name="SD3",
            encoders=[
                EncoderConfig(
                    name="clip_l",
                    tokenizer=pipeline.tokenizer,
                    text_encoder=pipeline.text_encoder,
                    seq_length=77,
                    vocab_size=len(pipeline.tokenizer.get_vocab()),
                    special_tokens={
                        "bos_token_id": pipeline.tokenizer.bos_token_id,
                        "eos_token_id": pipeline.tokenizer.eos_token_id,
                        "pad_token_id": pipeline.tokenizer.pad_token_id,
                    },
                    embedding_dim=768,
                ),
                EncoderConfig(
                    name="clip_g",
                    tokenizer=pipeline.tokenizer_2,
                    text_encoder=pipeline.text_encoder_2,
                    seq_length=77,
                    vocab_size=len(pipeline.tokenizer_2.get_vocab()),
                    special_tokens={
                        "bos_token_id": pipeline.tokenizer_2.bos_token_id,
                        "eos_token_id": pipeline.tokenizer_2.eos_token_id,
                        "pad_token_id": pipeline.tokenizer_2.pad_token_id,
                    },
                    embedding_dim=1280,
                ),
                EncoderConfig(
                    name="t5",
                    tokenizer=pipeline.tokenizer_3,
                    text_encoder=pipeline.text_encoder_3,
                    seq_length=256,  # Default T5 length
                    vocab_size=(
                        len(pipeline.tokenizer_3.get_vocab())
                        if pipeline.tokenizer_3
                        else 32100
                    ),
                    special_tokens={
                        "bos_token_id": (
                            getattr(pipeline.tokenizer_3, "bos_token_id", None)
                            if pipeline.tokenizer_3
                            else None
                        ),
                        "eos_token_id": (
                            getattr(pipeline.tokenizer_3, "eos_token_id", None)
                            if pipeline.tokenizer_3
                            else None
                        ),
                        "pad_token_id": (
                            getattr(pipeline.tokenizer_3, "pad_token_id", None)
                            if pipeline.tokenizer_3
                            else None
                        ),
                    },
                    embedding_dim=4096,
                ),
            ],
        )

    elif model_name.lower() == "sd15":
        return ModelConfig(
            model_name="SD1.5",
            encoders=[
                EncoderConfig(
                    name="clip",
                    tokenizer=pipeline.tokenizer,
                    text_encoder=pipeline.text_encoder,
                    seq_length=77,
                    vocab_size=len(pipeline.tokenizer.get_vocab()),
                    special_tokens={
                        "bos_token_id": pipeline.tokenizer.bos_token_id,
                        "eos_token_id": pipeline.tokenizer.eos_token_id,
                        "pad_token_id": pipeline.tokenizer.pad_token_id,
                    },
                    embedding_dim=768,
                )
            ],
        )

    elif model_name.lower() == "flux":
        return ModelConfig(
            model_name="Flux",
            encoders=[
                EncoderConfig(
                    name="clip_l",
                    tokenizer=pipeline.tokenizer,
                    text_encoder=pipeline.text_encoder,
                    seq_length=77,
                    vocab_size=len(pipeline.tokenizer.get_vocab()),
                    special_tokens={
                        "bos_token_id": pipeline.tokenizer.bos_token_id,
                        "eos_token_id": pipeline.tokenizer.eos_token_id,
                        "pad_token_id": pipeline.tokenizer.pad_token_id,
                    },
                    embedding_dim=768,
                ),
                EncoderConfig(
                    name="t5",
                    tokenizer=pipeline.tokenizer_2,
                    text_encoder=pipeline.text_encoder_2,
                    seq_length=512,  # Default T5 length
                    vocab_size=(
                        len(pipeline.tokenizer_2.get_vocab())
                        if pipeline.tokenizer_2
                        else 32100
                    ),
                    special_tokens={
                        "bos_token_id": (
                            getattr(pipeline.tokenizer_2, "bos_token_id", None)
                            if pipeline.tokenizer_2
                            else None
                        ),
                        "eos_token_id": (
                            getattr(pipeline.tokenizer_2, "eos_token_id", None)
                            if pipeline.tokenizer_2
                            else None
                        ),
                        "pad_token_id": (
                            getattr(pipeline.tokenizer_2, "pad_token_id", None)
                            if pipeline.tokenizer_2
                            else None
                        ),
                    },
                    embedding_dim=4096,
                ),
            ],
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def create_random_token_generator(
    pipeline, model_name: str, device: str = "cuda"
) -> RandomTokenGenerator:
    """Factory function: creates a random token generator for the corresponding model"""

    config = create_model_config(pipeline, model_name)

    if model_name.lower() == "sd3":
        return SD3RandomTokenGenerator(config, device)
    elif model_name.lower() == "flux":
        return FluxRandomTokenGenerator(config, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# =============================================================================
# ICG Plugin Interface - This is the main function called by the pipeline
# =============================================================================


def encode_prompt_with_icg(
    pipeline,
    pipeline_name: str,
    icg_generator: Optional[torch.Generator] = None,
    show_random_texts: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Main interface function for the ICG plugin

    Args:
        pipeline: SD3Pipeline instance
        icg_generator: ICG random number generator; if None, returns empty string embeddings (traditional CFG)
        show_random_texts: Whether to display the generated random texts

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (negative_prompt_embeds, negative_pooled_prompt_embeds)
    """

    # Get parameters from the pipeline's internal state
    # These parameters are already determined in the pipeline's __call__ method
    batch_size = 1  # Default to 1; if multiple prompts, handled in the pipeline
    num_images_per_prompt = 1  # Default to 1; can be adjusted via pipeline parameters

    # Check if ICG generator exists (i.e., if ICG is enabled)
    if icg_generator is None:
        # If no ICG generator, return embeddings for an empty string (traditional CFG)
        return _get_empty_prompt_embeddings(pipeline, batch_size, num_images_per_prompt)

    # Create ICG random token generator
    device = pipeline._execution_device
    icg_token_generator = create_random_token_generator(pipeline, pipeline_name, device)

    # Generate random embeddings
    result = icg_token_generator.generate_random_embeddings(
        batch_size=batch_size,
        num_images_per_prompt=num_images_per_prompt,
        generator=icg_generator,
        return_texts=show_random_texts,
    )

    if show_random_texts:
        (negative_prompt_embeds, negative_pooled_prompt_embeds), random_texts = result
        print("=" * 50)
        print("ICG Generated Random Texts:")
        for i, text in enumerate(random_texts):
            print(f"  {i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")
        print("=" * 50)
    else:
        negative_prompt_embeds, negative_pooled_prompt_embeds = result

    return negative_prompt_embeds, negative_pooled_prompt_embeds


def _get_empty_prompt_embeddings(
    pipeline, batch_size: int = 1, num_images_per_prompt: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets empty string prompt embeddings (for traditional CFG)
    """
    device = pipeline._execution_device

    # Call the appropriate encode_prompt method based on pipeline type
    if hasattr(pipeline, "tokenizer_3"):
        # SD3 model - has three encoders
        _, negative_prompt_embeds, _, negative_pooled_prompt_embeds = (
            pipeline.encode_prompt(
                prompt="",
                prompt_2="",
                prompt_3="",
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
            )
        )
    elif hasattr(pipeline, "tokenizer_2") and not hasattr(pipeline, "tokenizer_3"):
        # Flux format (returns 3 values)
        negative_prompt_embeds, negative_pooled_prompt_embeds, _ = (
            pipeline.encode_prompt(
                prompt="",
                prompt_2="",
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
        )
    else:
        # Single encoder model (SD1.5)
        negative_prompt_embeds, negative_pooled_prompt_embeds = (
            pipeline.encode_prompt(
                prompt="",
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
            )
        )

    # Handle batch dimension
    if batch_size > 1:
        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(
            batch_size, 1
        )

    return negative_prompt_embeds, negative_pooled_prompt_embeds
