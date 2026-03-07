import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from diffusers import FluxPipeline
from configs.utils import get_model_path
from tqdm import tqdm


def main(
    prompts, guidance_scale=1.5, num_inference_steps=28, seed=42, save_path_list=None
):
    """
    Call CFG method with FLUX model for text-to-image generation
    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (e.g., 1.5)
        num_inference_steps: Number of inference steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images
    """
    # Get model path from config
    model_path = get_model_path("flux")
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    gen_images = []

    # Output information
    print("Use CFG Flux model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Seed: {seed}")
    print("Note: Resolution set to 512x512 due to inference speed limitations")

    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images"
    ):
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt,
            height=512,  # Limited to 512 for faster inference speed
            width=512,  # Limited to 512 for faster inference speed
            negative_prompt="",
            true_cfg_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
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
        print(f"Image saved to: {save_path}/{i}.png")


if __name__ == "__main__":
    prompts = ["A man is cooking, MineCraft Style."]
    guidance_scale = 1.5
    num_inference_steps = 28
    seed = 195672
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    save_images(gen_images, "outputs/CFG/Flux")
    # CUDA_VISIBLE_DEVICES=3 python models/runner/cfg/flux.py
