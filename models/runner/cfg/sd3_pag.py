import torch
import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from diffusers import StableDiffusion3PAGPipeline
from configs.utils import get_model_path
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=7.5,
    num_inference_steps=28,
    seed=42,
    pag_applied_layers=None,
    pag_scale=0.7,
    save_path_list=None,
    debug=False,
):
    """
    Call CFG method for text-to-image generation using SD3 PAG model
    Args:
        prompts: List of prompt texts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (e.g., 7.5)
        num_inference_steps: Number of denoising steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        pag_applied_layers: PAG applied layers (e.g., ["blocks.13"])
        pag_scale: PAG scale (e.g., 0.7)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images (if save_path_list is None)
    """
    # Get model path from configuration
    model_path = get_model_path("sd3")

    # Set default PAG applied layers
    if pag_applied_layers is None:
        pag_applied_layers = ["blocks.13"]

    pipe = StableDiffusion3PAGPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, pag_applied_layers=pag_applied_layers
    )
    pipe.to("cuda")

    # Ensure inputs are in list format
    if isinstance(prompts, str):
        prompts = [prompts]

    if save_path_list is None:
        gen_images = []
    # Output information
    print("Use CFG SD3-PAG model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Pag scale: {pag_scale}")
    print(f"Pag applied layers: {pag_applied_layers}")
    print(f"Seed: {seed}")
    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images"
    ):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            pag_scale=pag_scale,
        ).images[0]
        if save_path_list is not None:
            image.save(save_path_list[i])
        else:
            gen_images.append(image)

    if save_path_list is None:
        return gen_images


def save_images(gen_images, save_path):
    import os

    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(gen_images):
        image.save(f"{save_path}/{i}.png")
        print(f"Save image to {save_path}/{i}.png")


if __name__ == "__main__":
    prompts = ["A cat holding a sign that says hello world."]
    guidance_scale = 7
    num_inference_steps = 28
    seed = 42
    pag_applied_layers = ["blocks.13"]
    pag_scale = 3
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        pag_applied_layers=pag_applied_layers,
        pag_scale=pag_scale,
    )
    save_images(gen_images, "outputs/CFG/SD3_PAG")

# CUDA_VISIBLE_DEVICES=2 python models/runner/cfg/sd3_pag.py
