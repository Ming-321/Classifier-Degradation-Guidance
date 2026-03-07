import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from models.pipelines.sfg.sd3.pipeline import StableDiffusion3SFGPipeline
from configs.utils import get_model_path
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=7,
    sfg_guidance_scale=2,
    sfg_scale=10.0,
    sfg_start_ratio=0.5,
    sfg_applied_layers_index=None,
    num_inference_steps=28,
    seed=42,
    save_path_list=None
):
    """
    Call SFG method with SD3 model for text-to-image generation
    
    Reference: "Segmentation-free guidance for text-to-image diffusion models" (CVPR 2024 Workshop)
    
    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: CFG guidance scale for Phase 1 (w in paper, default: 7)
        sfg_guidance_scale: SFG guidance scale for Phase 2 (w̄ in paper, default: 2)
        sfg_scale: Attention modification scale (a in paper, default: 10.0)
        sfg_start_ratio: Ratio to switch from CFG to SFG (default: 0.5 = T/2)
        sfg_applied_layers_index: List of transformer block indices to apply SFG (None = all layers)
        num_inference_steps: Number of inference steps (default: 28)
        seed: Random seed for image generation (default: 42)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png"])
    Returns:
        gen_images: Generated images (if save_path_list is None)
    """
    # Get model path from config
    model_path = get_model_path("sd3")
    pipe = StableDiffusion3SFGPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    pipe.to("cuda")

    gen_images = []

    # Output information
    print("Use SFG SD3 model to generate images")
    print(f"Guidance scale (CFG phase): {guidance_scale}")
    print(f"SFG guidance scale (SFG phase): {sfg_guidance_scale}")
    print(f"SFG scale (a): {sfg_scale}")
    print(f"SFG start ratio: {sfg_start_ratio}")
    print(f"SFG applied layers: {sfg_applied_layers_index if sfg_applied_layers_index else 'all'}")
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
            sfg_guidance_scale=sfg_guidance_scale,
            sfg_scale=sfg_scale,
            sfg_start_ratio=sfg_start_ratio,
            sfg_applied_layers_index=sfg_applied_layers_index,
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
    
    # SFG parameters (paper defaults adapted for SD3)
    guidance_scale = 7        # w: CFG guidance for phase 1
    sfg_guidance_scale = 2    # w̄: SFG guidance for phase 2
    sfg_scale = 10.0          # a: attention modification scale
    sfg_start_ratio = 0.5     # t_s/T: switch point (0.5 = T/2)
    sfg_applied_layers_index = None  # Apply to all layers
    
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        sfg_guidance_scale=sfg_guidance_scale,
        sfg_scale=sfg_scale,
        sfg_start_ratio=sfg_start_ratio,
        sfg_applied_layers_index=sfg_applied_layers_index,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    save_images(gen_images, "outputs/SFG/SD3")
    # CUDA_VISIBLE_DEVICES=0 python models/runner/sfg/sd3.py
