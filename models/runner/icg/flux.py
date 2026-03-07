import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from models.pipelines.icg.flux.pipeline import ICGFluxPipeline
from configs.utils import get_model_path
import torch
import random
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=2.5,
    num_inference_steps=28,
    seed=42,
    icg_seed=42,
    show_random_texts=False,
    save_path_list=None,
):
    """
    Call ICG method with Flux model for text-to-image generation

    Args:
        prompts: Input prompts list (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (e.g., 2.5)
        num_inference_steps: Number of inference steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        icg_seed: Random seed for ICG random number generator for generating random texts (e.g., 42)
        show_random_texts: Whether to show ICG generated random texts (e.g., False)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])

    Returns:
        gen_images: Generated images list
    """
    # Get model path from config
    model_path = get_model_path("flux")
    pipe = ICGFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Initialize ICG random number generator
    if icg_seed is not None:
        icg_random = random.Random(icg_seed)
    else:
        icg_random = None

    gen_images = []

    # Output information
    print("Use ICG Flux model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Seed: {seed}")
    print(f"ICG seed: {icg_seed}")
    print(f"Show random texts: {show_random_texts}")
    print("Note: Resolution set to 512x512 due to inference speed limitations")

    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images"
    ):

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Generate random icg_seed for this prompt if ICG is enabled
        current_icg_seed = None
        if icg_random is not None:
            current_icg_seed = icg_random.randint(0, 2**32 - 1)
            print(f"ICG seed: {current_icg_seed}")
        # Call ICG pipeline
        image = pipe(
            prompt=prompt,
            height=512,  # Limited to 512 for faster inference speed
            width=512,  # Limited to 512 for faster inference speed
            negative_prompt="",
            generator=generator,
            true_cfg_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            icg_seed=current_icg_seed,
            show_random_texts=show_random_texts,
        ).images[0]

        if save_path_list is not None:
            image.save(save_path_list[i])
        else:
            gen_images.append(image)

    if save_path_list is None:
        return gen_images


def save_images(gen_images, save_path):
    """
    Save generated images to specified path

    Args:
        gen_images: Generated images list
        save_path: Save path
    """
    import os

    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(gen_images):
        image.save(f"{save_path}/{i}.png")
        print(f"Image saved to: {save_path}/{i}.png")


if __name__ == "__main__":
    # Basic test
    prompts = ["A man is cooking, MineCraft Style."]
    guidance_scale = 1.5
    num_inference_steps = 28
    seed = 42
    icg_seed = 42
    show_random_texts = False

    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        icg_seed=icg_seed,
        show_random_texts=show_random_texts,
    )

    # Save images
    save_images(gen_images, "outputs/ICG/Flux")

    # Usage example:
    # CUDA_VISIBLE_DEVICES=3 python models/runner/icg/flux.py
