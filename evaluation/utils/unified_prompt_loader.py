#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Prompt Loader - Integrates loading logic for all prompt sources
"""

import os
import json
import random
import argparse
from typing import List, Tuple, Optional, Dict, Any

# Use environment check tool to set up path
from .env_check import setup_project_path

setup_project_path()


class UnifiedPromptLoader:
    """Unified Prompt Loader - Handles all prompt sources"""

    @staticmethod
    def load_prompts_from_args(args: argparse.Namespace) -> List[Tuple[str, Optional[str]]]:
        """
        Load prompts based on command line arguments

        Args:
            args: Command line arguments object, should contain the following attributes:
                - prompt_data_file: JSON data file path (optional)
                - prompt_file: Text file path (optional)
                - prompts: Directly specified prompt list (optional)
                - max_prompts: Maximum prompt count (optional)
                - prompt_seed: Random seed (optional, default 2025)

        Returns:
            List[Tuple[str, Optional[str]]]: Prompt data list, each element is (prompt, image_id)

        Raises:
            ValueError: If no prompt source is specified
            FileNotFoundError: If specified file doesn't exist

        Design principles:
        1. Priority: prompt_data_file > prompt_file > prompts > COCO default
        2. Unified format: All sources return (prompt, image_id) format
        3. Friendly errors: Provide clear error messages and suggestions
        """
        prompt_data = []

        # Get arguments, provide default values
        max_prompts = getattr(args, "max_prompts", None)
        prompt_seed = getattr(args, "prompt_seed", 2025)

        # Try to load prompts by priority
        if hasattr(args, "prompt_data_file") and args.prompt_data_file:
            # 1. Load from JSON data file
            prompt_data = UnifiedPromptLoader._load_from_json_file(args.prompt_data_file, max_prompts, prompt_seed)
            print(f"Loaded {len(prompt_data)} prompts from JSON file: {args.prompt_data_file}")

        elif hasattr(args, "prompt_file") and args.prompt_file:
            # 2. Load from text file
            prompt_data = UnifiedPromptLoader._load_from_text_file(args.prompt_file, max_prompts, prompt_seed)
            print(f"Loaded {len(prompt_data)} prompts from text file: {args.prompt_file}")

        elif hasattr(args, "prompts") and args.prompts:
            # 3. Directly specified from command line arguments
            prompt_data = UnifiedPromptLoader._load_from_list(args.prompts, max_prompts, prompt_seed)
            print(f"Loaded {len(prompt_data)} prompts from command line arguments")

        else:
            # 4. Use COCO default data
            prompt_data = UnifiedPromptLoader._load_from_coco_default(max_prompts, prompt_seed)
            print(f"Loaded {len(prompt_data)} prompts from COCO default data")

        if not prompt_data:
            raise ValueError(
                "Cannot load any prompts! Please check one of the following parameters:\n"
                "  --prompt_data_file <JSON file path>\n"
                "  --prompt_file <text file path>\n"
                "  --prompts <directly specified prompts>\n"
                "or ensure COCO default configuration is correct"
            )

        return prompt_data

    @staticmethod
    def _load_from_json_file(
        file_path: str, max_prompts: Optional[int] = None, random_seed: int = 2025
    ) -> List[Tuple[str, Optional[str]]]:
        """Load prompt data from JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file does not exist: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support multiple JSON formats
            if isinstance(data, list):
                # Format 1: [{"prompt": "...", "image_id": "..."}, ...]
                if all(isinstance(item, dict) and "prompt" in item for item in data):
                    prompt_data = [
                        (item["prompt"], str(item.get("image_id")) if item.get("image_id") is not None else None)
                        for item in data
                    ]
                # Format 2: [["prompt1", "id1"], ["prompt2", "id2"], ...]
                elif all(isinstance(item, list) and len(item) >= 2 for item in data):
                    prompt_data = [(item[0], str(item[1]) if item[1] is not None else None) for item in data]
                # Format 3: ["prompt1", "prompt2", ...]
                elif all(isinstance(item, str) for item in data):
                    prompt_data = [(item, None) for item in data]
                else:
                    raise ValueError(f"Unsupported JSON format: {file_path}")
            else:
                raise ValueError(f"JSON file should contain a list, not {type(data).__name__}")

            # Apply max_prompts limit
            if max_prompts and len(prompt_data) > max_prompts:
                random.seed(random_seed)
                prompt_data = random.sample(prompt_data, max_prompts)

            return prompt_data

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON file format error: {file_path}\nError: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {file_path}\nError: {e}")

    @staticmethod
    def _load_from_text_file(
        file_path: str, max_prompts: Optional[int] = None, random_seed: int = 2025
    ) -> List[Tuple[str, Optional[str]]]:
        """Load prompts from text file (one per line)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file does not exist: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]

            # Filter empty lines
            prompts = [line for line in lines if line]

            if not prompts:
                raise ValueError(f"Text file is empty or contains only empty lines: {file_path}")

            # Apply max_prompts limit
            if max_prompts and len(prompts) > max_prompts:
                random.seed(random_seed)
                prompts = random.sample(prompts, max_prompts)

            # Convert to unified format
            prompt_data = [(prompt, None) for prompt in prompts]
            return prompt_data

        except Exception as e:
            raise ValueError(f"Failed to load text file: {file_path}\nError: {e}")

    @staticmethod
    def _load_from_list(
        prompt_list: List[str], max_prompts: Optional[int] = None, random_seed: int = 2025
    ) -> List[Tuple[str, Optional[str]]]:
        """Load from prompt list"""
        if not prompt_list:
            raise ValueError("Prompt list is empty")

        # Apply max_prompts limit
        if max_prompts and len(prompt_list) > max_prompts:
            random.seed(random_seed)
            prompt_list = random.sample(prompt_list, max_prompts)

        # Convert to unified format
        prompt_data = [(prompt, None) for prompt in prompt_list]
        return prompt_data

    @staticmethod
    def _load_from_coco_default(
        max_prompts: Optional[int] = None, random_seed: int = 2025
    ) -> List[Tuple[str, Optional[str]]]:
        """Load prompts from COCO default configuration"""
        try:
            from configs.utils import get_prompt_source_config

            prompt_config = get_prompt_source_config()
            coco_annotation_file = prompt_config["coco_annotation_file"]

            if not coco_annotation_file:
                raise ValueError("COCO annotation file path not configured")

            if not os.path.exists(coco_annotation_file):
                raise FileNotFoundError(f"COCO annotation file does not exist: {coco_annotation_file}")

            print(f"Using COCO annotation file: {coco_annotation_file}")

            # Directly implement COCO loading logic
            prompt_data = UnifiedPromptLoader._load_from_coco_annotations(
                coco_annotation_file, max_prompts=max_prompts, random_seed=random_seed
            )

            return prompt_data

        except ImportError:
            raise ValueError("Cannot import configs.utils module, please check configuration")
        except Exception as e:
            raise ValueError(f"Failed to load from COCO default configuration: {e}")

    @staticmethod
    def _load_from_coco_annotations(
        coco_annotation_file: str, max_prompts: Optional[int] = None, random_seed: int = 2025
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Load prompts from COCO annotation file

        Args:
            annotation_file: COCO annotation file path
            max_prompts: Maximum prompt count
            random_seed: Random seed

        Returns:
            List[Tuple[str, Optional[str]]]: (prompt, image_id) tuple list
        """
        with open(coco_annotation_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Extract annotation information, only select first prompt for each image
        image_prompts = {}
        for annotation in coco_data.get("annotations", []):
            if "caption" in annotation and "image_id" in annotation:
                image_id = str(annotation["image_id"])
                # Only add when image_id hasn't been added yet (ensure only first prompt per image)
                if image_id not in image_prompts:
                    image_prompts[image_id] = annotation["caption"]

        # Convert to list format
        prompt_data = [(caption, image_id) for image_id, caption in image_prompts.items()]

        if max_prompts and len(prompt_data) > max_prompts:
            random.seed(random_seed)
            prompt_data = random.sample(prompt_data, max_prompts)

        return prompt_data

    @staticmethod
    def validate_prompt_data(prompt_data: List[Tuple[str, Optional[str]]]) -> bool:
        """
        Validate if prompt data format is correct

        Args:
            prompt_data: Prompt data list

        Returns:
            bool: Returns True if format is correct, otherwise False
        """
        if not isinstance(prompt_data, list):
            return False

        for item in prompt_data:
            if not isinstance(item, tuple) or len(item) != 2:
                return False

            prompt, image_id = item
            if not isinstance(prompt, str) or not prompt.strip():
                return False

            if image_id is not None and not isinstance(image_id, str):
                return False

        return True

    @staticmethod
    def get_prompts_only(prompt_data: List[Tuple[str, Optional[str]]]) -> List[str]:
        """
        Extract pure prompt list from prompt data

        Args:
            prompt_data: Prompt data list

        Returns:
            List[str]: Pure prompt list
        """
        return [prompt for prompt, _ in prompt_data]

    @staticmethod
    def get_usage_help() -> str:
        """Get usage help information"""
        return """
Prompt Loader Usage Instructions:

1. Load from JSON file (highest priority):
   --prompt_data_file <file path>
   
   Supported JSON formats:
   - [{"prompt": "description", "image_id": "ID"}, ...]
   - [["description1", "ID1"], ["description2", "ID2"], ...]
   - ["description1", "description2", ...]

2. Load from text file:
   --prompt_file <file path>
   
   Format: One prompt per line

3. Directly specify from command line:
   --prompts "Prompt1" "Prompt2" ...

4. Use COCO default data (last choice):
   Automatically used when no prompt source is specified

Common Args:
   --max_prompts <count>     Limit maximum prompt count
   --prompt_seed <seed>      Random sampling seed (default 2025)

Examples:
   python -m evaluation.generate --model cfg/sd3 --config config.json --prompt_file prompts.txt --max_prompts 100
   python -m evaluation.generate --model cfg/sd3 --config config.json --prompts "a cat" "a dog" --output_dir results
        """


# Convenience functions
def load_prompts_from_args(args: argparse.Namespace) -> List[Tuple[str, Optional[str]]]:
    """
    Convenience function for loading prompts from command line arguments

    Args:
        args: Command line arguments object

    Returns:
        List[Tuple[str, Optional[str]]]: Prompt data list
    """
    return UnifiedPromptLoader.load_prompts_from_args(args)


def get_prompts_only(prompt_data: List[Tuple[str, Optional[str]]]) -> List[str]:
    """
    Convenience function for extracting pure prompt list from prompt data

    Args:
        prompt_data: Prompt data list

    Returns:
        List[str]: Pure prompt list
    """
    return UnifiedPromptLoader.get_prompts_only(prompt_data)


if __name__ == "__main__":
    # Test code
    print("=== Prompt Loader Test ===")

    # Create test arguments
    class TestArgs:
        def __init__(self):
            self.prompts = ["a beautiful cat", "a dog in the park"]
            self.max_prompts = None
            self.prompt_seed = 2025

    args = TestArgs()

    try:
        prompt_data = load_prompts_from_args(args)
        print(f"✅ Successfully loaded {len(prompt_data)} prompts")

        for i, (prompt, image_id) in enumerate(prompt_data):
            print(f"  {i+1}. '{prompt}' (ID: {image_id})")

        # Test validation
        is_valid = UnifiedPromptLoader.validate_prompt_data(prompt_data)
        print(f"✅ Data format validation: {'Passed' if is_valid else 'Failed'}")

        # Test extracting pure prompts
        prompts_only = get_prompts_only(prompt_data)
        print(f"✅ Extracted pure prompts: {prompts_only}")

    except Exception as e:
        print(f"❌ Test failed: {e}")

    print("\n=== Usage Help ===")
    print(UnifiedPromptLoader.get_usage_help())
