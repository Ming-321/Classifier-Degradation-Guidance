import torch
import random
import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from models.pipelines.cdg.sd3.pipeline import CDGSD3Img2ImgPipeline
from diffusers.utils import load_image
from configs.utils import get_model_path
from tqdm import tqdm


def main(
    prompts,
    init_images,
    strength,
    guidance_scale,
    process_params,
    seed=42,
    debug=False,
    save_path_list=None,
):
    """
    Call CDG method for image-to-image generation using SD3 model
    Args:
        prompts: List of prompt texts (e.g., ["A man is cooking, MineCraft Style."])
        init_images: List of initial image paths (e.g., ["input1.png", "input2.png"])
        strength: Transformation strength (e.g., 0.6)
        guidance_scale: Guidance scale (e.g., 7.5)
        process_params: Processing parameters (e.g., {"process_index": 1, "keep_ratio": {"content":0,"padding":1}, ...})
        seed: Random seed for image generation (e.g., 42)
        debug: Whether to enable debug mode (e.g., False)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images (if save_path_list is None)
    """
    # Get model path from configuration
    model_path = get_model_path("sd3")
    pipe = CDGSD3Img2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")

    # Ensure inputs are in list format
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(init_images, str):
        init_images = [init_images]

    # Expand init_images list to match the length of prompts
    if len(init_images) == 1 and len(prompts) > 1:
        init_images = init_images * len(prompts)

    if save_path_list is None:
        gen_images = []

    # Set Python random seed once before the loop for reproducible random ordering
    # Each image will have different random token ordering, but the sequence is reproducible
    random.seed(seed)

    for i, (prompt, init_image_path) in tqdm(
        enumerate(zip(prompts, init_images)),
        total=len(prompts),
        desc="Generating images",
    ):
        # Set PyTorch generator seed for reproducible image noise
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
            process_params=process_params,
            debug=debug,
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
    process_params = {
        "process_index": 3,
        "degrade_ratio": {"content": 0.8, "padding": 0}
    }
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
        process_params=process_params,
        seed=seed,
    )
    save_images(gen_images, "outputs/CDG/SD3_img2img")
    # CUDA_VISIBLE_DEVICES=2 python models/runner/cdg/sd3_img2img.py
