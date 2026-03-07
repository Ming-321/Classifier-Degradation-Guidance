import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from models.pipelines.cdg.sd3.pipeline import CDGSD3Pipeline
from configs.utils import get_model_path
import torch
import random
from tqdm import tqdm



def main(
    prompts,
    guidance_scale,
    num_inference_steps,
    process_params,
    seed=42,
    debug=False,
    save_path_list=None,
):
    """
    Call CDG method with SD3 model for text-to-image generation
    Args:
        prompts: Input prompts (e.g., ["A man is cooking, MineCraft Style."])
        guidance_scale: Guidance scale (e.g., 7)
        num_inference_steps: Number of inference steps (e.g., 28)
        process_params: Processing parameters (e.g., {"process_index": 1, "keep_ratio": {"content":0,"padding":1}, "separate_clip_t5": True, "calculate_params": {...}})
        seed: Random seed for image generation (e.g., 42)
        debug: Enable debug mode (e.g., False)
        save_path_list: List of save paths (e.g., ["path/to/output/0.png", "path/to/output/1.png"])
    Returns:
        gen_images: Generated images
    """
    # Get model path from config
    model_path = get_model_path("sd3")
    pipe = CDGSD3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")

    gen_images = []

    # Output information
    print("Use CDG SD3 model to generate images")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Num inference steps: {num_inference_steps}")
    print(f"Process params: {process_params}")
    print(f"Seed: {seed}")

    # Set Python random seed once before the loop for reproducible random ordering
    # Each image will have different random token ordering, but the sequence is reproducible
    random.seed(seed)

    for i, prompt in tqdm(
        enumerate(prompts), total=len(prompts), desc="Generating images"
    ):
        # Set PyTorch generator seed for reproducible image noise
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
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
        print(f"Image saved to: {save_path}/{i}.png")


if __name__ == "__main__":
    # Parameter descriptions:
    # process_index: int, Processing index indicating which transformer block to process in each denoising step.
    #                SD3 model has 28 transformer blocks, so process_index ranges from 0-27.
    # degrade_ratio: dict, Degrade ratio determining which tokens to degrade and their ratios, works with separate_clip_t5.
    #             If {"all":k}, degrade all tokens with ratio k; if {"content":k1,"padding":k2}, degrade content and padding tokens with ratios k1 and k2 respectively. These two are mutually exclusive.
    process_params = {
        "process_index": 1,
        "degrade_ratio": {"content": 1, "padding": 0},
        "separate_clip_t5": True,
        "use_negative_pooled_prompt_embeds": False,
        "all_use_first_step_importance": True,
        "use_random_sorted_indices": False,
        "calculate_params": {
            "cal_score_algorithm": "text_only_pagerank",
            "cal_sorted_indices_algorithm": "variance_weighted",
            "process_type": "min",
            "v_min": 0.00,
            "v_max": None,
            "epsilon": 1e-4,
            "max_iterations": 20,
            "norm_type": "L1",
            "damping_factor": 0.0
        }
    }
    debug = False
    prompts = ["A man is cooking, MineCraft Style."]
    guidance_scale = 7
    num_inference_steps = 28
    seed = 42
    gen_images = main(
        prompts=prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        process_params=process_params,
        seed=seed,
        debug=debug,
    )
    save_images(gen_images, "outputs/CDG/SD3")
    # CUDA_VISIBLE_DEVICES=0 python models/runner/cdg/sd3.py
