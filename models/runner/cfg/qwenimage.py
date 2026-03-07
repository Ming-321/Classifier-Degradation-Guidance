import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from models.pipelines.cdg.qwenimage.pipeline import CDGQwenImagePipeline
from configs.utils import get_model_path, get_method_config
import torch
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=4.0,
    num_inference_steps=25,
    seed=42,
    save_path_list=None,
):
    """
    Call CFG method with Qwen-Image model for text-to-image generation (baseline).

    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (mapped to true_cfg_scale for Qwen-Image)
        num_inference_steps: Number of inference steps (e.g., 25)
        seed: Random seed for image generation (e.g., 42)
        save_path_list: List of save paths
    Returns:
        gen_images: Generated images (if save_path_list is None)
    """
    # Get model path from config
    model_path = get_model_path("qwenimage")

    # Read method config for device settings
    try:
        config = get_method_config("cfg", "qwenimage")
    except FileNotFoundError:
        config = {}

    transformer_gpu = config.get("transformer_gpu", 0)
    text_encoder_gpu = config.get("text_encoder_gpu", 1)
    vae_gpu = config.get("vae_gpu", 2)
    negative_prompt = config.get("negative_prompt", " ")
    width = config.get("width", 512)
    height = config.get("height", 512)
    max_sequence_length = config.get("max_sequence_length", 512)

    # Load pipeline
    cfg_pipe = CDGQwenImagePipeline.from_pretrained(
        model_path=model_path,
        transformer_gpu=transformer_gpu,
        text_encoder_gpu=text_encoder_gpu,
        vae_gpu=vae_gpu,
        torch_dtype=torch.bfloat16,
    )

    gen_images = []

    # Output information
    print("Use CFG Qwen-Image model to generate images (baseline)")
    print(f"Guidance scale (true_cfg_scale): {guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Resolution: {width}x{height}")
    print(f"Seed: {seed}")
    print(f"GPUs: transformer={transformer_gpu}, text_encoder={text_encoder_gpu}, vae={vae_gpu}")

    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Generating images"):
        current_seed = seed + i

        image = cfg_pipe.generate_cfg_baseline(
            prompt=prompt,
            seed=current_seed,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance_scale,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
        )

        if save_path_list is not None:
            os.makedirs(os.path.dirname(save_path_list[i]), exist_ok=True)
            image.save(save_path_list[i])
        else:
            gen_images.append(image)

    if save_path_list is None:
        return gen_images


def save_images(gen_images, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(gen_images):
        image.save(f"{save_path}/{i}.png")
        print(f"Image saved to: {save_path}/{i}.png")


if __name__ == "__main__":
    prompts = ["A man is cooking, MineCraft Style."]
    guidance_scale = 4.0
    num_inference_steps = 25
    seed = 42
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    save_images(gen_images, "outputs/CFG/QwenImage")
    # CUDA_VISIBLE_DEVICES=0,1,2 python models/runner/cfg/qwenimage.py
