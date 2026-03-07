import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from models.pipelines.cads.sd3.pipeline import CADSSD3Pipeline
from configs.utils import get_model_path
import torch
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=7,
    num_inference_steps=28,
    seed=42,
    s=0.07,
    save_path_list=None,
):
    """
    Call CADS method with SD3 model for text-to-image generation

    Args:
        prompts: Input prompts list (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (e.g., 7)
        num_inference_steps: Number of inference steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        s: CADS noise intensity parameter (e.g., 0.07)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])

    Returns:
        gen_images: Generated images list
    """
    # Get model path from config
    model_path = get_model_path("sd3")
    pipe = CADSSD3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")

    gen_images = []

    # Output information
    print("Use CADS SD3 model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Random seed: {seed}")
    print(f"CADS noise intensity (s): {s}")

    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images"
    ):

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Call CADS pipeline
        image = pipe(
            prompt=prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            s=s,  # CADS noise intensity parameter
            mu=None,  # Use default dynamic shifting
        ).images[0]

        if save_path_list is not None:
            image.save(save_path_list[i])
            print(f"Image saved to: {save_path_list[i]}")
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
    guidance_scale = 7
    num_inference_steps = 28
    seed = 42
    s = 0.07  # CADS noise intensity

    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        s=s,
    )

    # Save images
    save_images(gen_images, "outputs/CADS/SD3")

    # CUDA_VISIBLE_DEVICES=3 python models/runner/cads/sd3.py
