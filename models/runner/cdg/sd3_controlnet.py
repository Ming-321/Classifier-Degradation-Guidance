import torch
import random
import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from models.pipelines.cdg.sd3.pipeline import CDGSD3ControlNetPipeline
from diffusers.models import SD3ControlNetModel
from diffusers.utils import load_image
from configs.utils import get_model_path
from tqdm import tqdm


def main(
    prompts,
    control_images,
    process_params,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=28,
    guidance_scale=5.5,
    seed=42,
    save_path_list=None,
):
    """
    Call CDG method for text-to-image generation using SD3 ControlNet model
    Args:
        prompts: List of prompt texts (e.g., ["A girl wearing a suit..."])
        control_images: List of control image paths (e.g., ["canny1.jpg", "canny2.jpg"])
        process_params: Processing parameters (e.g., {"process_index": 1, "keep_ratio": {"content":0,"padding":1}, ...})
        negative_prompts: List of negative prompts (e.g., ["NSFW, nude..."])
        controlnet_conditioning_scale: ControlNet conditioning scale (e.g., 0.5)
        num_inference_steps: Number of denoising steps (e.g., 28)
        guidance_scale: Guidance scale (e.g., 5.5)
        seed: Random seed for image generation (e.g., 42)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images (if save_path_list is None)
    """
    # Get model paths from configuration
    model_path = get_model_path("sd3")
    controlnet_model_path = get_model_path("sd3_controlnet")

    # load pipeline
    controlnet = SD3ControlNetModel.from_pretrained(controlnet_model_path,torch_dtype=torch.float16)
    pipe = CDGSD3ControlNetPipeline.from_pretrained(
        model_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.to("cuda")

    # Ensure input lists have consistent lengths
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(control_images, str):
        control_images = [control_images]

    # Expand lists to match the length of prompts
    if len(control_images) == 1 and len(prompts) > 1:
        control_images = control_images * len(prompts)

    if save_path_list is None:
        gen_images = []

    # Set Python random seed once before the loop for reproducible random ordering
    # Each image will have different random token ordering, but the sequence is reproducible
    random.seed(seed)

    for i, (prompt, control_image_path) in tqdm(
        enumerate(zip(prompts, control_images)),
        total=len(prompts),
        desc="Generating images",
    ):
        # Set PyTorch generator seed for reproducible image noise
        generator = torch.Generator(device="cuda").manual_seed(seed)
        control_image = load_image(control_image_path)

        image = pipe(
            prompt,
            negative_prompt="",
            control_image=control_image,
            process_params=process_params,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
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
        print(f"Saved image {i} to {save_path}/{i}.png")


if __name__ == "__main__":
    control_images = ["figures/canny.png"]
    prompts = [
        "A two-color racing car drives through a damp city street at night, its front is green and the rear is black, and the number '77' is written on the door. A blue neon sign on the side of the road reads 'DREAM'."
    ]
    controlnet_conditioning_scale = 0.5
    num_inference_steps = 28
    guidance_scale = 7
    seed = 195672
    process_params = {
        "process_index": 3,
        "degrade_ratio": {"content": 1, "padding": 0.2}
    }
    gen_images = main(
        prompts,
        control_images,
        process_params,
        controlnet_conditioning_scale,
        num_inference_steps,
        guidance_scale,
        seed,
    )
    save_images(gen_images, "outputs/CDG/SD3_ControlNet")

# CUDA_VISIBLE_DEVICES=2 python models/runner/cdg/sd3_controlnet.py
