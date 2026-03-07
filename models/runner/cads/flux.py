import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from models.pipelines.cads.flux.pipeline import CADSFluxPipeline
from configs.utils import get_model_path
import torch
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=2.5,
    num_inference_steps=28,
    seed=42,
    s=0.07,
    save_path_list=None,
):
    """
    Call CADS method with Flux model for text-to-image generation

    Args:
        prompts: Input prompts list (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (e.g., 2.5)
        num_inference_steps: Number of inference steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        s: CADS noise intensity parameter (e.g., 0.07)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])

    Returns:
        gen_images: Generated images list
    """
    # Get model path from config
    model_path = get_model_path("flux")
    pipe = CADSFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    gen_images = []

    # Output information
    print("Use CADS Flux model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Seed: {seed}")
    print(f"CADS noise scale (s): {s}")
    print("Note: Resolution set to 512x512 due to inference speed limitations")

    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images"
    ):

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Call CADS pipeline
        image = pipe(
            prompt=prompt,
            height=512,  # Limited to 512 for faster inference speed
            width=512,  # Limited to 512 for faster inference speed
            negative_prompt="",
            generator=generator,
            true_cfg_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            s=s,  # CADS noise intensity parameter
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
    s = 0.03  # CADS noise intensity (aligned with COCO-Flux experiment configuration)

    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        s=s,
    )

    # Save images
    save_images(gen_images, "outputs/CADS/Flux")

    # Usage example:
    # CUDA_VISIBLE_DEVICES=3 python models/runner/cads/flux.py
