#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple

# Use environment check tool to set up path
from .utils.env_check import setup_project_path

setup_project_path()

# Now can use absolute imports
from evaluation.core.model_interface import create_model_runner
from evaluation.utils.model_parser import parse_model_string
from evaluation.utils.unified_prompt_loader import load_prompts_from_args, get_prompts_only
from evaluation.utils.filename_utils import sanitize_filename


# parse_model_string function has been moved to unified model parser


def create_image_mappings(
    prompt_data: List[Tuple[str, Optional[str]]], images_dir: str
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Create image filename mappings and save paths.

    Args:
        prompt_data: Prompt data list [(prompt, image_id), ...]
        images_dir: Image save directory

    Returns:
        Tuple[List[str], List[Dict[str, Any]]]: (save path list, mapping info list)
    """
    save_path_list = []
    mappings = []

    for i, (prompt, image_id) in enumerate(prompt_data):
        # Generate filename
        if image_id:
            filename = f"{image_id}.png"
        else:
            safe_filename = sanitize_filename(prompt)
            filename = f"{safe_filename}_{i:04d}.png"

        # Save path
        image_path = os.path.join(images_dir, filename)
        save_path_list.append(image_path)

        # Create mapping information
        mapping = {
            "prompt": prompt,
            "image": filename,
            "image_path": os.path.join("images", filename),  # Relative path
            "absolute_image_path": image_path,  # Absolute path
        }
        if image_id:
            mapping["image_id"] = image_id

        mappings.append(mapping)

    return save_path_list, mappings


def save_generation_metadata(
    output_dir: str, args: argparse.Namespace, prompts: List[str], config_data: Dict[str, Any]
) -> None:
    """
    Save generation metadata.

    Args:
        output_dir: Output directory
        args: Command line arguments
        prompts: List of prompts
        config_data: Configuration data
    """
    metadata = {
        "model": args.model,
        "config_file": args.config,
        "num_prompts": len(prompts),
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "config_data": config_data,
    }

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"Generation metadata saved to {metadata_file}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Refactored image generation script")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier, format: model_type or model_type/model_variant, e.g. cfg or cfg/flux",
    )
    parser.add_argument("--config", type=str, required=True, help="JSON configuration file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on, e.g. cuda, cuda:0, cpu, etc.")

    # Prompt-related parameters
    prompt_group = parser.add_argument_group("Prompts")
    prompt_group.add_argument(
        "--prompt_data_file", type=str, help="JSON file path containing (prompt, image_id) tuple list"
    )
    prompt_group.add_argument("--prompt_file", type=str, help="File path containing prompts, one per line")
    prompt_group.add_argument("--prompts", type=str, nargs="+", help="Directly specified list of prompts")
    prompt_group.add_argument("--max_prompts", type=int, default=None, help="Maximum number of prompts to process")
    prompt_group.add_argument("--prompt_seed", type=int, default=2025, help="Seed for random prompt sampling")

    # Generation parameters
    gen_group = parser.add_argument_group("Generation parameters")
    gen_group.add_argument("--output_dir", type=str, required=True, help="Output directory")
    gen_group.add_argument(
        "--temp_json_filename", type=str, default="prompts.json", help="JSON filename for saving prompt mappings"
    )
    gen_group.add_argument("--seed", type=int, default=42, help="Random seed for image generation")

    args = parser.parse_args()

    # Validate parameters - new config system will automatically handle config file lookup

    # Parse model type and variant (using unified strict mode parser)
    try:
        model_type, model_variant = parse_model_string(args.model, "generate.py")
        print(f"Using model: {model_type}/{model_variant}")
    except ValueError as e:
        parser.error(str(e))

    # Load configuration
    try:
        from configs.utils import load_and_validate_config

        config_dict = load_and_validate_config(args.config, model_type, model_variant)
    except Exception as e:
        parser.error(f"Failed to load configuration: {e}")

    # Load prompts (using unified prompt loader)
    try:
        prompt_data = load_prompts_from_args(args)
        prompts = get_prompts_only(prompt_data)
        print(f"Loaded {len(prompts)} prompts")
    except Exception as e:
        parser.error(f"Failed to load prompts: {e}")

    # Ensure output directory and images subdirectory exist
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Create model runner, pass in configuration
    try:
        model_runner = create_model_runner(model_type, model_variant, seed=args.seed, config=config_dict)
        print(f"Successfully created {model_type}/{model_variant} model runner")
    except Exception as e:
        print(f"Failed to create model runner: {e}")
        return

    # Pre-create save paths and mapping information
    save_path_list, mappings = create_image_mappings(prompt_data, images_dir)

    # Generate and save images simultaneously
    print(f"Starting image generation, will save in real-time to: {images_dir}")
    start_time = time.time()

    try:
        # Use generate-and-save interface
        model_runner.generate_and_save(prompts, save_path_list)
        generation_time = time.time() - start_time
        print(f"Image generation and saving completed, took {generation_time:.2f} seconds")

    except Exception as e:
        print(f"Error generating images: {e}")
        return

    # Save mapping file
    temp_json_path = os.path.join(args.output_dir, args.temp_json_filename)
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=4, ensure_ascii=False)

    print(f"Prompt mapping saved to {temp_json_path}")

    # Save metadata
    save_generation_metadata(args.output_dir, args, prompts, config_dict)

    print(f"Generation complete! Saved {len(save_path_list)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
