#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Model Parser - Only supports strict mode, but provides friendly error messages
"""
import sys
import os
from typing import Tuple

# Use environment check tool to set up path
from .env_check import setup_project_path

setup_project_path()

# Now can use absolute imports
from configs.utils import get_available_methods, validate_model_method_combination


class UnifiedModelParser:
    """Unified Model Parser - Strict mode with friendly error messages"""

    @staticmethod
    def parse_model_string(model_string: str, context: str = "") -> Tuple[str, str]:
        """
        Parse model string, strictly require format as method/model

        Args:
            model_string: Model string, must be in 'method/model' format, such as 'cfg/sd3'
            context: Context information, used for error messages (such as script name)

        Returns:
            Tuple[str, str]: (method_name, model_name) tuple
        """
        context_info = f" (from {context})" if context else ""

        # Check basic format
        if "/" not in model_string:
            return UnifiedModelParser._handle_missing_variant_error(model_string, context_info)

        # Parse method and model
        parts = model_string.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model format: '{model_string}'{context_info}")

        method_name, model_name = parts[0].strip(), parts[1].strip()

        # Validate that method and model are not empty
        if not method_name or not model_name:
            return UnifiedModelParser._handle_empty_parts_error(model_string, context_info)

        # Validate if combination is valid
        if not validate_model_method_combination(method_name, model_name):
            return UnifiedModelParser._handle_invalid_combination_error(method_name, model_name, context_info)

        return method_name, model_name

    @staticmethod
    def _handle_missing_variant_error(model_string: str, context_info: str) -> None:
        """Handle missing variant error, provide friendly help information"""
        available_methods = get_available_methods()

        if model_string.strip() in available_methods:
            # User entered a valid method but didn't specify model
            method_name = model_string.strip()
            available_models = available_methods[method_name]

            # Generate usage examples
            examples = [f"{method_name}/{model}" for model in available_models[:3]]
            examples_text = "、".join(examples)

            raise ValueError(
                f"Model format error{context_info}: Must specify specific model variant\n"
                f"\n"
                f"Your input method '{method_name}' supports the following models:\n"
                f"  {available_models}\n"
                f"\n"
                f"Please use complete format: {examples_text}\n"
                f"Recommended: {method_name}/{available_models[0]}"
            )
        else:
            # User entered an invalid method
            available_list = list(available_methods.keys())

            raise ValueError(
                f"Model format error{context_info}: Must use 'method/model' format\n"
                f"\n"
                f"Available methods: {available_list}\n"
                f"\n"
                f"Usage examples:\n"
                f"  cfg/sd3     - Use CFG method with SD3 model\n"
                f"  cdg/flux    - Use CDG method with Flux model\n"
                f"  cdg/sd35    - Use CDG method with SD3.5 model\n"
                f"\n"
                f"Please specify complete format, such as: cfg/sd3"
            )

    @staticmethod
    def _handle_empty_parts_error(model_string: str, context_info: str) -> None:
        """Handle empty parts error"""
        raise ValueError(
            f"Model format error{context_info}: '{model_string}' contains empty parts\n"
            f"\n"
            f"Correct format: method/model\n"
            f"Examples: cfg/sd3, cdg/flux, cdg/sd35"
        )

    @staticmethod
    def _handle_invalid_combination_error(method_name: str, model_name: str, context_info: str) -> None:
        """Handle invalid combination error, provide available options"""
        available_methods = get_available_methods()

        if method_name in available_methods:
            # Method exists but model is not supported
            available_models = available_methods[method_name]
            suggestions = [f"{method_name}/{model}" for model in available_models]

            raise ValueError(
                f"Unsupported combination{context_info}: {method_name}/{model_name}\n"
                f"\n"
                f"Method '{method_name}' supports the following models:\n"
                f"  {available_models}\n"
                f"\n"
                f"Please use one of the following combinations:\n"
                f"  {chr(10).join('  ' + s for s in suggestions)}\n"
                f"\n"
                f"Recommended: {suggestions[0]}"
            )
        else:
            # Method doesn't exist
            available_list = list(available_methods.keys())
            examples = []
            for method in available_list[:3]:
                models = available_methods[method]
                examples.append(f"{method}/{models[0]}")

            raise ValueError(
                f"Unknown method{context_info}: '{method_name}'\n"
                f"\n"
                f"Available methods: {available_list}\n"
                f"\n"
                f"Usage examples:\n"
                f"  {chr(10).join('  ' + ex for ex in examples)}\n"
                f"\n"
                f"Please check method name spelling"
            )

    @staticmethod
    def get_available_combinations() -> dict:
        """
        Get all available method/model combinations

        Returns:
            Dict[str, list]: method -> [model1, model2, ...] mapping
        """
        return get_available_methods()

    @staticmethod
    def validate_combination(method_name: str, model_name: str) -> bool:
        """
        Validate if method/model combination is valid

        Args:
            method_name: Method name
            model_name: Model name

        Returns:
            bool: Returns True if combination is valid, otherwise False
        """
        return validate_model_method_combination(method_name, model_name)


# Convenience functions
def parse_model_string(model_string: str, context: str = "") -> Tuple[str, str]:
    """
    Convenience function for parsing model string

    Args:
        model_string: Model string in 'method/model' format
        context: Context information for error messages

    Returns:
        Tuple[str, str]: (method_name, model_name) tuple

    Raises:
        ValueError: If format is incorrect or unsupported combination
    """
    return UnifiedModelParser.parse_model_string(model_string, context)


def get_usage_help() -> str:
    """
    Get usage help information

    Returns:
        str: Detailed usage help text
    """
    available_methods = get_available_methods()

    help_text = ["Model parameter usage instructions:", "", "Format: --model method/model", "", "Available combinations:"]

    for method, models in available_methods.items():
        help_text.append(f"  {method}:")
        for model in models:
            help_text.append(f"    {method}/{model}")
        help_text.append("")

    help_text.extend(
        [
            "Usage examples:",
            "  python -m evaluation.generate --model cfg/sd3 ...",
            "  python -m evaluation.generate_distributed --model cdg/flux ...",
            "",
            "Note: Must use complete 'method/model' format",
        ]
    )

    return "\n".join(help_text)


if __name__ == "__main__":
    # Test code
    print("=== Model Parser Test ===")

    test_cases = [
        "cfg/sd3",  # Correct format
        "cdg",  # Missing model
        "padding/test",  # Invalid combination
        "cfg/",  # Empty model
        "/sd3",  # Empty method
    ]

    for test in test_cases:
        try:
            result = parse_model_string(test, "test")
            print(f"✅ '{test}' -> {result}")
        except ValueError as e:
            print(f"❌ '{test}' -> {e}")
        print()

    print("=== Usage Help ===")
    print(get_usage_help())
