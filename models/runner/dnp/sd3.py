"""
DNP (Diffusion-Negative Prompting) Runner for SD3

Paper: "Improving image synthesis with diffusion-negative sampling" (Desai & Vasconcelos, ECCV 2024)

DNP Algorithm:
1. DNS (Diffusion-Negative Sampling): Generate image least compliant with prompt p
   - Implementation: pipe(prompt="", negative_prompt=p)
2. Caption the DNS image using BLIP2 to get negative prompt n*
3. Final generation: pipe(prompt=p, negative_prompt=n*)
"""

import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch
import json
from diffusers import StableDiffusion3Pipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from configs.utils import get_model_path
from tqdm import tqdm
from typing import List, Optional
from PIL import Image


def load_blip2_model(model_path: str, device: str = "cuda"):
    """
    Load BLIP2 model for image captioning
    
    Args:
        model_path: Path to BLIP2 model
        device: Device to load model on
        
    Returns:
        processor: BLIP2 processor
        model: BLIP2 model
    """
    print(f"Loading BLIP2 model from {model_path}...")
    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map=device
    )
    print("BLIP2 model loaded successfully")
    return processor, model


def caption_image(image: Image.Image, processor, model, max_new_tokens: int = 50) -> str:
    """
    Generate caption for an image using BLIP2
    
    Args:
        image: PIL Image to caption
        processor: BLIP2 processor
        model: BLIP2 model
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        caption: Generated caption string
    """
    inputs = processor(images=image, return_tensors="pt").to(model.device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def main(
    prompts: List[str],
    guidance_scale: float = 7,
    num_inference_steps: int = 28,
    seed: int = 42,
    dns_guidance_scale: Optional[float] = None,
    debug: bool = False,
    blip2_model_path: str = "Salesforce/blip2-opt-2.7b",
    save_path_list: Optional[List[str]] = None,
) -> Optional[List[Image.Image]]:
    """
    DNP method with SD3 model for text-to-image generation
    
    Args:
        prompts: Input prompts
        guidance_scale: Guidance scale for final generation
        num_inference_steps: Number of inference steps
        seed: Random seed for image generation
        dns_guidance_scale: Guidance scale for DNS stage (defaults to guidance_scale if None)
        blip2_model_path: Path to BLIP2 model for captioning (default: placeholder path)
        debug: Enable debug mode to save intermediate results
        save_path_list: List of save paths for final images
        
    Returns:
        gen_images: Generated images (if save_path_list is None)
    """
    # Use same guidance scale for DNS if not specified
    if dns_guidance_scale is None:
        dns_guidance_scale = guidance_scale
    
    # Get model path from config
    model_path = get_model_path("sd3")
    
    # Load SD3 pipeline
    print("Loading SD3 model...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    pipe.to("cuda")
    print("SD3 model loaded successfully")
    
    # Load BLIP2 model
    blip2_processor, blip2_model = load_blip2_model(blip2_model_path)
    
    gen_images = []
    
    # Output information
    print("=" * 50)
    print("DNP (Diffusion-Negative Prompting) Image Generation")
    print("=" * 50)
    print(f"Guidance scale: {guidance_scale}")
    print(f"DNS guidance scale: {dns_guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Seed: {seed}")
    print(f"Debug mode: {debug}")
    print("=" * 50)
    
    # Debug output directory
    if debug:
        debug_dir = "outputs/DNP/SD3/debug"
        os.makedirs(debug_dir, exist_ok=True)
        debug_metadata = []
    
    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images (DNP)"
    ):
        # Create generator with seed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # ============================================
        # Step 1: DNS (Diffusion-Negative Sampling)
        # Generate image least compliant with prompt
        # Implementation: positive="", negative=prompt
        # ============================================
        print(f"\n[{i+1}/{len(prompts)}] Processing: {prompt[:50]}...")
        print("  Step 1: DNS - Generating negative image...")
        
        dns_generator = torch.Generator(device="cuda").manual_seed(seed)
        dns_image = pipe(
            prompt="",
            negative_prompt=prompt,
            guidance_scale=dns_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=dns_generator,
        ).images[0]
        
        if debug:
            dns_save_path = os.path.join(debug_dir, f"dns_image_{i}.png")
            dns_image.save(dns_save_path)
            print(f"  DNS image saved to: {dns_save_path}")
        
        # ============================================
        # Step 2: Caption DNS image using BLIP2
        # ============================================
        print("  Step 2: Captioning DNS image with BLIP2...")
        negative_prompt = caption_image(dns_image, blip2_processor, blip2_model)
        print(f"  Generated negative prompt: {negative_prompt}")
        
        # ============================================
        # Step 3: Final generation with (prompt, negative_prompt)
        # ============================================
        print("  Step 3: Final generation with DNP...")
        
        final_generator = torch.Generator(device="cuda").manual_seed(seed)
        final_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=final_generator,
        ).images[0]
        
        # Save or collect image
        if save_path_list is not None:
            final_image.save(save_path_list[i])
            print(f"  Final image saved to: {save_path_list[i]}")
        else:
            gen_images.append(final_image)
        
        # Debug metadata
        if debug:
            debug_metadata.append({
                "index": i,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale,
                "dns_guidance_scale": dns_guidance_scale,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
            })
    
    # Save debug metadata
    if debug:
        metadata_path = os.path.join(debug_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(debug_metadata, f, indent=2, ensure_ascii=False)
        print(f"\nDebug metadata saved to: {metadata_path}")
    
    if save_path_list is None:
        return gen_images


def save_images(gen_images: List[Image.Image], save_path: str):
    """
    Save generated images to directory
    
    Args:
        gen_images: List of PIL Images
        save_path: Directory to save images
    """
    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(gen_images):
        image.save(f"{save_path}/{i}.png")
        print(f"Image saved to: {save_path}/{i}.png")


if __name__ == "__main__":
    # Test prompts
    prompts = ["A man is cooking, MineCraft Style."]
    
    # Parameters
    guidance_scale = 7
    num_inference_steps = 28
    seed = 42
    debug = False
    
    # Run DNP
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        debug=debug,
    )
    
    # Save final images
    save_images(gen_images, "outputs/DNP/SD3")
    
    # Command to run:
    # CUDA_VISIBLE_DEVICES=0 python models/runner/dnp/sd3.py
