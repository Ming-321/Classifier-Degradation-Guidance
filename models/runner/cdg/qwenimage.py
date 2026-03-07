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
    process_params=None,
    seed=42,
    debug=False,
    save_path_list=None,
):
    """
    Call CDG method with Qwen-Image model for text-to-image generation.

    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (mapped to true_cfg_scale for Qwen-Image)
        num_inference_steps: Number of inference steps (e.g., 25)
        process_params: Processing parameters, e.g.:
            {
                "tail_n": 5,
                "tail_variant": "tail_only",
                "process_index": -1,
                "positive_magic": "",
                "max_sequence_length": 512,
                "width": 512,
                "height": 512,
                "transformer_gpu": 0,
                "text_encoder_gpu": 1,
                "vae_gpu": 2,
            }
        seed: Random seed for image generation (e.g., 42)
        debug: Enable debug mode (e.g., False)
        save_path_list: List of save paths
    Returns:
        gen_images: Generated images (if save_path_list is None)
    """
    if process_params is None:
        process_params = {}

    # Get model path from config
    model_path = get_model_path("qwenimage")

    # Extract device configuration from process_params
    transformer_gpu = process_params.get("transformer_gpu", 0)
    text_encoder_gpu = process_params.get("text_encoder_gpu", 1)
    vae_gpu = process_params.get("vae_gpu", 2)

    # Extract CDG parameters
    tail_n = process_params.get("tail_n", 5)
    tail_variant = process_params.get("tail_variant", "tail_only")
    process_index = process_params.get("process_index", -1)
    positive_magic = process_params.get("positive_magic", "")
    max_sequence_length = process_params.get("max_sequence_length", 512)
    width = process_params.get("width", 512)
    height = process_params.get("height", 512)

    # Load pipeline
    cdg_pipe = CDGQwenImagePipeline.from_pretrained(
        model_path=model_path,
        transformer_gpu=transformer_gpu,
        text_encoder_gpu=text_encoder_gpu,
        vae_gpu=vae_gpu,
        torch_dtype=torch.bfloat16,
    )

    gen_images = []

    # Output information
    print("Use CDG Qwen-Image model to generate images")
    print(f"Guidance scale (true_cfg_scale): {guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Tail N: {tail_n}, Tail variant: {tail_variant}, Process index: {process_index}")
    print(f"Resolution: {width}x{height}")
    print(f"Seed: {seed}")
    print(f"GPUs: transformer={transformer_gpu}, text_encoder={text_encoder_gpu}, vae={vae_gpu}")

    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Generating images"):
        current_seed = seed + i

        image = cdg_pipe.generate_cdg(
            prompt=prompt,
            seed=current_seed,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance_scale,
            tail_n=tail_n,
            tail_variant=tail_variant,
            process_index=process_index,
            positive_magic=positive_magic,
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
    # Read configuration from config file
    config = get_method_config("cdg", "qwenimage")

    guidance_scale = config.get("guidance_scale", 4.0)
    num_inference_steps = config.get("num_inference_steps", 25)
    seed = config.get("seed", 42)
    debug = config.get("debug", False)
    process_params = config.get("process_params", {})

    prompts = ["A man is cooking, MineCraft Style."]

    print("=" * 60)
    print("CDG Qwen-Image Runner")
    print("=" * 60)

    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        process_params=process_params,
        seed=seed,
        debug=debug,
    )
    save_images(gen_images, "outputs/CDG/QwenImage")
    # CUDA_VISIBLE_DEVICES=0,1,2 python models/runner/cdg/qwenimage.py
