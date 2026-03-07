import torch
import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from configs.utils import get_model_path
from tqdm import tqdm


def main(prompts, init_images, strength, guidance_scale, seed=42, save_path_list=None):
    """
    Call CFG method with SD3 model for img2img generation
    Args:
        prompts: List of prompts (e.g., ["A man is cooking, MineCraft Style."])
        init_images: List of initial image paths (e.g., ["input1.png", "input2.png"])
        strength: Strength parameter (e.g., 0.6)
        guidance_scale: Guidance scale (e.g., 7)
        seed: Random seed for image generation (e.g., 42)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images (if save_path_list is None)
    """
    # Get model path from config
    model_path = get_model_path("sd3")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    pipe.to("cuda")

    # Ensure inputs are in list format
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(init_images, str):
        init_images = [init_images]

    # Expand init_images list to match prompts length
    if len(init_images) == 1 and len(prompts) > 1:
        init_images = init_images * len(prompts)

    if save_path_list is None:
        gen_images = []

    for i, (prompt, init_image_path) in tqdm(
        enumerate(zip(prompts, init_images)),
        total=len(prompts),
        desc="Generating images",
    ):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        init_image = (
            load_image(init_image_path)
            if isinstance(init_image_path, str)
            else init_image_path
        )
        image = pipe(
            prompt,
            generator=generator,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
        ).images[0]
        if save_path_list is not None:
            image.save(save_path_list[i])
        else:
            gen_images.append(image)

    if save_path_list is None:
        return gen_images


def save_images(gen_images, save_path):
    """
    Save generated images to specified directory.

    Args:
        gen_images: List of generated images
        save_path: Directory path to save images
    """
    import os

    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(gen_images):
        image.save(f"{save_path}/{i}.png")
        print(f"Save image to {save_path}/{i}.png")


if __name__ == "__main__":
    init_images = ["figures/init.png"]
    prompts = ["A golden retriever was sitting on the grass, surrounded by three children playing, and in the distance was a forest."]
    strength = 0.8
    guidance_scale = 7
    seed = 19926
    gen_images = main(
        prompts=prompts,
        init_images=init_images,
        strength=strength,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    save_images(gen_images, "outputs/CFG/SD3_img2img")
    # CUDA_VISIBLE_DEVICES=2 python models/runner/cfg/sd3_img2img.py
