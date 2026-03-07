import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from diffusers import StableDiffusion3PAGPipeline
from configs.utils import get_model_path
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=1.0,
    pag_scale=3.0,
    pag_applied_layers=None,
    num_inference_steps=28,
    seed=42,
    save_path_list=None,
):
    """
    Call PAG method with SD3.5 model for text-to-image generation
    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (e.g., 1.0)
        pag_scale: PAG scale (e.g., 3.0). Set to 0 to disable PAG.
        pag_applied_layers: List of transformer blocks to apply PAG (e.g., ["blocks.13"])
        num_inference_steps: Number of inference steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images
    """
    # Get model path from config
    model_path = get_model_path("sd35")
    
    # Set default PAG applied layers if not specified
    if pag_applied_layers is None:
        pag_applied_layers = ["blocks.13"]
    
    pipe = StableDiffusion3PAGPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        pag_applied_layers=pag_applied_layers
    )
    pipe.to("cuda")

    gen_images = []

    # Output information
    print("Use PAG SD3.5 model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"PAG scale: {pag_scale}")
    print(f"PAG applied layers: {pag_applied_layers}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Seed: {seed}")

    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images"
    ):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            pag_scale=pag_scale,
            num_inference_steps=num_inference_steps,
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
    prompts = ["A cat holding a sign that says hello world"]
    num_inference_steps = 28
    seed = 42
    
    # Test mode 1: Pure PAG (guidance_scale=1.0)
    # guidance_scale = 1.0
    # pag_scale = 3.0
    
    # Test mode 2: Pure CFG (pag_scale=0)
    # guidance_scale = 3.5
    # pag_scale = 0
    
    # Test mode 3: CFG + PAG
    guidance_scale = 1.0
    pag_scale = 3.0
    
    # PAG parameters
    pag_applied_layers = ["blocks.13"]  # Apply to block 13. Can use multiple: ["blocks.12", "blocks.13"]
    
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        pag_scale=pag_scale,
        pag_applied_layers=pag_applied_layers,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    save_images(gen_images, "outputs/PAG/SD3.5")
    # CUDA_VISIBLE_DEVICES=3 python models/runner/pag/sd35.py




