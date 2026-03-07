#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

# Use environment check tool to set up path
from .utils.env_check import setup_project_path

setup_project_path()

# Now can use absolute imports
from evaluation.utils.task_manager import TaskManager
from evaluation.utils.model_parser import parse_model_string
from evaluation.utils.unified_prompt_loader import load_prompts_from_args


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run distributed image generation on multiple GPUs (with resume support)"
    )

    # Basic parameters
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier, format: model_type/model_variant, e.g. cfg/sd3 or cdg/flux",
    )
    parser.add_argument("--config", type=str, required=True, help="JSON configuration file path")
    parser.add_argument("--gpus", type=str, required=True, help="GPU IDs to use, comma-separated, e.g. 0,1,2")

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
    gen_group.add_argument("--seed", type=int, default=42, help="Random seed for image generation")

    # Output parameters
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory")

    # Task management parameters
    task_group = parser.add_argument_group("Task management")
    task_group.add_argument(
        "--resume", action="store_true", default=True, help="Enable resume functionality (enabled by default)"
    )
    task_group.add_argument("--no_resume", action="store_true", help="Disable resume functionality, start from scratch")
    task_group.add_argument("--status", action="store_true", help="View task status without running")

    args = parser.parse_args()

    # Handle resume parameter
    if args.no_resume:
        args.resume = False

    # Validate parameters - new config system will automatically handle config file lookup

    # Parse GPU IDs
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",")]
    print(f"Will use {len(gpu_ids)} GPUs: {gpu_ids}")

    # Create task manager
    task_manager = TaskManager(args.output_dir)

    # If only viewing status
    if args.status:
        status = task_manager.get_status()
        print("Task status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return

    # Parse model string (using unified strict mode parser)
    try:
        model_type, model_variant = parse_model_string(args.model, "generate_distributed.py")
        print(f"Using model: {model_type}/{model_variant}")
    except ValueError as e:
        parser.error(str(e))

    # Load prompts (using unified prompt loader)
    try:
        prompt_data = load_prompts_from_args(args)
        print(f"Total loaded {len(prompt_data)} prompts")
    except Exception as e:
        parser.error(f"Failed to load prompts: {e}")

    # Run distributed generation
    try:
        task_manager.run_distributed_generation(
            model_type=model_type,
            model_variant=model_variant,
            config_path=args.config,
            prompt_data=prompt_data,
            gpu_ids=gpu_ids,
            resume=args.resume,
            seed=args.seed,
        )

        print(f"\nDistributed generation completed! Results saved in: {args.output_dir}")
        print(f"Can run evaluation script:")
        print(
            f"python evaluation/eval.py --generated_path '{args.output_dir}' --metrics FID,CLIPScore --batch_size 250"
        )

    except Exception as e:
        print(f"Distributed generation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
