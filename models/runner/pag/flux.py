import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from models.pipelines.pag.flux.pipeline import PAGFluxPipeline
from configs.utils import get_model_path
from tqdm import tqdm


def main(
    prompts,
    guidance_scale=3.5,
    pag_scale=3.0,
    pag_applied_layers=None,
    pag_adaptive_scale=0.0,
    true_cfg_scale=1.0,
    num_inference_steps=28,
    seed=42,
    save_path_list=None,
):
    """
    Call PAG method with Flux model for text-to-image generation
    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale for Flux guidance embedding (e.g., 3.5)
        pag_scale: PAG scale (e.g., 3.0). Set to 0 to disable PAG.
        pag_applied_layers: List of transformer layers to apply PAG (e.g., ["transformer_blocks.10"])
        pag_adaptive_scale: Adaptive scale factor for PAG (default: 0.0)
        true_cfg_scale: True CFG scale for negative prompts (e.g., 3.0). Set to 1.0 to disable CFG.
        num_inference_steps: Number of inference steps (e.g., 28)
        seed: Random seed for image generation (e.g., 42)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images
    """
    # Get model path from config
    model_path = get_model_path("flux")
    
    # Set default PAG applied layers if not specified
    if pag_applied_layers is None:
        pag_applied_layers = ["transformer_blocks.10"]
    
    pipe = PAGFluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        pag_applied_layers=pag_applied_layers
    )
    pipe.to("cuda")

    gen_images = []

    # Output information
    print("Use PAG Flux model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"PAG scale: {pag_scale}")
    print(f"PAG applied layers: {pag_applied_layers}")
    print(f"PAG adaptive scale: {pag_adaptive_scale}")
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
            pag_scale=pag_scale,
            pag_adaptive_scale=pag_adaptive_scale,
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
    
    # Test mode 1: Pure PAG (true_cfg_scale=1.0)
    # true_cfg_scale = 1.0
    # pag_scale = 3.0
    
    # Test mode 2: Pure CFG (pag_scale=0)
    # true_cfg_scale = 3.0
    # pag_scale = 0
    
    # Test mode 3: CFG + PAG (recommended)
    true_cfg_scale = 3.0
    pag_scale = 3.0
    
    # PAG parameters
    guidance_scale = 3.5
    pag_applied_layers = ["transformer_blocks.10"]  # Apply to block 10. Can use multiple: ["transformer_blocks.8", "transformer_blocks.10"]
    pag_adaptive_scale = 0.0  # Set > 0 for adaptive scaling
    
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        pag_scale=pag_scale,
        pag_applied_layers=pag_applied_layers,
        pag_adaptive_scale=pag_adaptive_scale,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    save_images(gen_images, "outputs/PAG/Flux")
    # CUDA_VISIBLE_DEVICES=0 python models/runner/pag/flux.py

