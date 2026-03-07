import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from models.pipelines.icg.sd3.pipeline import ICGSD3Pipeline
from configs.utils import get_model_path
import torch
import random
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=7,
    num_inference_steps=28,
    seed=42,
    icg_seed=42,
    show_random_texts=False,
    save_path_list=None,
):
    """
    Call ICG method with SD3 model for text-to-image generation
    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (e.g., 7)
        num_inference_steps: Number of inference steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        icg_seed: Random seed for ICG random number generator, if None then use traditional CFG (e.g., 42)
        show_random_texts: Whether to show random texts (e.g., True)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images
    """
    # Get model path from config
    model_path = get_model_path("sd3")
    pipe = ICGSD3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")

    # Initialize ICG random number generator
    if icg_seed is not None:
        icg_random = random.Random(icg_seed)
    else:
        icg_random = None

    gen_images = []

    # Output information
    print("Use ICG SD3 model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Random seed: {seed}")
    print(
        f"ICG seed: {icg_seed} ({'ICG enabled' if icg_seed is not None else 'Use traditional CFG'})"
    )
    print(f"Show random texts: {show_random_texts}")

    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images"
    ):

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Generate random icg_seed for this prompt if ICG is enabled
        current_icg_seed = None
        if icg_random is not None:
            current_icg_seed = icg_random.randint(0, 2**32 - 1)
            print(f"ICG seed: {current_icg_seed}")

        image = pipe(
            prompt=prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            icg_seed=current_icg_seed,
            show_random_texts=show_random_texts,
        ).images[0]

        if save_path_list is not None:
            image.save(save_path_list[i])
            print(f"Image saved to: {save_path_list[i]}")
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
    guidance_scale = 7
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
    save_images(gen_images, "outputs/ICG/SD3")
    # CUDA_VISIBLE_DEVICES=3 python models/runner/icg/sd3.py
