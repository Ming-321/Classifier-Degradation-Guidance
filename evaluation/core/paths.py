#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path


class ProjectPaths:
    """Project path manager, unified management of all paths"""

    def __init__(self):
        # Get project root directory (one level up from evaluation directory)
        self.project_root = Path(__file__).parent.parent.parent
        self.evaluation_root = self.project_root / "evaluation"

    @property
    def models_root(self):
        return self.project_root / "models"

    @property
    def outputs_root(self):
        return self.project_root / "outputs"

    @property
    def configs_root(self):
        return self.project_root / "configs"

    def get_model_runner_path(self, model_type: str, model_variant: str):
        """Get model runner path"""
        return self.models_root / "runner" / model_type / f"{model_variant}.py"

    def get_config_path(self, config_file: str):
        """
        Get absolute path of configuration file, supports multiple search methods.

        Args:
            config_file: Configuration file path

        Returns:
            Path: Absolute path of configuration file

        Raises:
            FileNotFoundError: If configuration file does not exist
        """
        if Path(config_file).is_absolute():
            return Path(config_file)

        # First search from root configs directory
        config_path = self.configs_root / config_file
        if config_path.exists():
            return config_path

        # Try searching from methods subdirectory
        methods_path = self.configs_root / "methods" / config_file
        if methods_path.exists():
            return methods_path

        # Try searching from evaluations subdirectory
        evaluations_path = self.configs_root / "evaluations" / config_file
        if evaluations_path.exists():
            return evaluations_path

        # Fallback: project root directory
        root_config_path = self.project_root / config_file
        if root_config_path.exists():
            return root_config_path

        raise FileNotFoundError(f"Configuration file does not exist: {config_file}")


# Global path management instance
paths = ProjectPaths()
