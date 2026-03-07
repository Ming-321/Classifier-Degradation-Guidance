import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from models.pipelines.seg.flux.pipeline import SEGFluxPipeline
from configs.utils import get_model_path
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=3.5,
    seg_scale=3.0,
    seg_blur_sigma=9999999.0,
    seg_applied_layers_index=None,
    true_cfg_scale=1.0,
    num_inference_steps=28,
    seed=42,
    save_path_list=None,
):
    """
    Call SEG method with Flux model for text-to-image generation
    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale for Flux guidance embedding (e.g., 3.5)
        seg_scale: SEG scale (e.g., 3.0). Set to 0 to disable SEG.
        seg_blur_sigma: Sigma value for Gaussian blur. Values > 9999.0 use uniform blur (default: 9999999.0)
        seg_applied_layers_index: List of transformer block indices to apply SEG (e.g., [0] for first block)
        true_cfg_scale: True CFG scale for negative prompts (e.g., 3.0). Set to 1.0 to disable CFG.
        num_inference_steps: Number of inference steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images
    """
    # Get model path from config
    model_path = get_model_path("flux")
    pipe = SEGFluxPipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    gen_images = []

    # Output information
    print("Use SEG Flux model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"SEG scale: {seg_scale}")
    print(f"SEG blur sigma: {seg_blur_sigma}")
    print(f"SEG applied layers: {seg_applied_layers_index if seg_applied_layers_index else [0]}")
    print(f"True CFG scale: {true_cfg_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Seed: {seed}")
    print("Note: Resolution set to 512x512 due to inference speed limitations")

    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images"
    ):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Determine if we need negative prompt based on true_cfg_scale
        negative_prompt = "" if true_cfg_scale > 1.0 else None
        
        image = pipe(
            prompt,
            height=512,  # Limited to 512 for faster inference speed
            width=512,  # Limited to 512 for faster inference speed
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            seg_scale=seg_scale,
            seg_blur_sigma=seg_blur_sigma,
            seg_applied_layers_index=seg_applied_layers_index,
            num_inference_steps=num_inference_steps,
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
        print(f"Image saved to: {save_path}/{i}.png")


if __name__ == "__main__":
    prompts = ["A cat holding a sign that says hello world"]
    num_inference_steps = 28
    seed = 42
    
    # Test mode 1: Pure SEG (true_cfg_scale=1.0)
    # true_cfg_scale = 1.0
    # seg_scale = 3.0
    
    # Test mode 2: Pure CFG (seg_scale=0)
    # true_cfg_scale = 3.0
    # seg_scale = 0
    
    # Test mode 3: CFG + SEG (recommended)
    true_cfg_scale = 3.0
    seg_scale = 3.0
    
    # SEG parameters
    guidance_scale = 3.5
    seg_blur_sigma = 9999999.0  # Infinite blur (uniform). For Gaussian blur, use smaller values like 1.0-3.0
    seg_applied_layers_index = [0]  # Apply SEG to first transformer block. Can use multiple: [0, 1, 2]
    
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        seg_scale=seg_scale,
        seg_blur_sigma=seg_blur_sigma,
        seg_applied_layers_index=seg_applied_layers_index,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    save_images(gen_images, "outputs/SEG/Flux")
    # CUDA_VISIBLE_DEVICES=0 python models/runner/seg/flux.py

