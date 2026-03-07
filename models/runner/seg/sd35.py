import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from models.pipelines.seg.sd3.pipeline import StableDiffusion3SEGPipeline
from configs.utils import get_model_path
from tqdm import tqdm


def main(
    prompts, 
    guidance_scale=1.0,
    seg_scale=0.75, 
    seg_blur_sigma=9999999.0,
    seg_applied_layers_index=None,
    num_inference_steps=28, 
    seed=42, 
    save_path_list=None
):
    """
    Call SEG method with SD3.5 model for text-to-image generation
    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (e.g., 1.0)
        seg_scale: SEG scale (e.g., 0.75)
        seg_blur_sigma: Sigma value for Gaussian blur. Values > 9999.0 use uniform blur (default: 9999999.0)
        seg_applied_layers_index: List of transformer block indices to apply SEG (e.g., [0] for first block)
        num_inference_steps: Number of inference steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images
    """
    # Get model path from config
    model_path = get_model_path("sd35")
    pipe = StableDiffusion3SEGPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    gen_images = []

    # Output information
    print("Use SEG SD3.5 model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"SEG scale: {seg_scale}")
    print(f"SEG blur sigma: {seg_blur_sigma}")
    print(f"SEG applied layers: {seg_applied_layers_index if seg_applied_layers_index else [0]}")
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
            seg_scale=seg_scale,
            seg_blur_sigma=seg_blur_sigma,
            seg_applied_layers_index=seg_applied_layers_index,
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
    prompts = ["A man is cooking, MineCraft Style."]
    num_inference_steps = 28
    seed = 42
    
    # Test mode 1: Pure CFG (seg_scale=0)
    # guidance_scale = 3.5
    # seg_scale = 0
    
    # Test mode 2: Pure SEG (guidance_scale=1.0)
    # Aligns with experiment configuration: guidance_scale=1, seg_scale=3, sigma=5, lambda_block=13
    guidance_scale = 1
    seg_scale = 3
    
    # Test mode 3: CFG + SEG
    # guidance_scale = 3.5
    # seg_scale = 3
    
    # SEG parameters (aligned with paper Appendix Table C.1)
    seg_blur_sigma = 5  # Gaussian blur sigma (paper: sigma=5)
    seg_applied_layers_index = [13]  # Apply SEG to block 13 (paper: lambda_block=13)
    
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        seg_scale=seg_scale,
        seg_blur_sigma=seg_blur_sigma,
        seg_applied_layers_index=seg_applied_layers_index,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    save_images(gen_images, "outputs/SEG/SD3.5")
    # CUDA_VISIBLE_DEVICES=3 python models/runner/seg/sd35.py




