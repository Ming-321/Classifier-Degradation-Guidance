#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from PIL import Image
import importlib.util
import sys
import os
import inspect

# Use environment check tool to set up path
from ..utils.env_check import setup_project_path

setup_project_path()

# Now can use absolute imports
from configs.utils import get_model_path, get_method_config


class BaseModelRunner(ABC):
    """Base model runner interface"""

    def __init__(self, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.model_variant = model_variant
        self.seed = seed  # Used to control randomness in generated images
        self.config = config  # External configuration passed in

    @abstractmethod
    def generate_and_save(self, prompts: List[str], save_path_list: List[str]) -> None:
        """
        Generate and save images simultaneously (must be implemented).

        Args:
            prompts: List of prompts
            save_path_list: List of save paths, should have same length as prompts
        """
        pass

    def _get_config(self) -> Dict[str, Any]:
        """Get configuration, prioritize external config, otherwise use system default config"""
        if self.config is not None:
            print(f"Using external configuration: guidance_scale={self.config.get('guidance_scale', 'N/A')}")
            return self.config
        else:
            # Use system default configuration (loaded from configs directory)
            print(f"Using system default configuration: {self.model_type}/{self.model_variant}")
            return get_method_config(self.model_type, self.model_variant)


class CFGModelRunner(BaseModelRunner):
    """CFG model runner"""

    def __init__(self, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, model_variant, seed, config)
        self.run_function = self._load_run_function()

    def _load_run_function(self):
        """Load the main function of corresponding runner script"""
        # Build module path
        module_path = f"models.runner.{self.model_type}.{self.model_variant}"

        try:
            # Dynamically import module
            module = importlib.import_module(module_path)

            if not hasattr(module, "main"):
                raise AttributeError(f"Module {module_path} does not have main function")

            return module.main
        except ImportError:
            raise ImportError(f"Cannot import module: {module_path}")

    def generate_and_save(self, prompts: List[str], save_path_list: List[str]) -> None:
        """
        CFG model's generate-and-save implementation.
        Uses CFG model's save_path_list parameter for true real-time saving.
        """
        if len(prompts) != len(save_path_list):
            raise ValueError(
                f"Number of prompts({len(prompts)}) doesn't match number of save paths({len(save_path_list)})"
            )

        # Ensure all save directories exist
        for save_path in save_path_list:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get configuration (prioritize external configuration)
        config = self._get_config()

        guidance_scale = config.get("guidance_scale", 7.5)
        num_inference_steps = config.get("num_inference_steps", 28)

        print(
            f"CFG model using configuration: guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}"
        )

        # Prepare basic parameters
        kwargs = {
            "prompts": prompts,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": self.seed,
            "save_path_list": save_path_list,
        }

        # Only add PAG parameters for PAG models
        if "pag" in self.model_variant.lower():
            kwargs["pag_scale"] = config.get("pag_scale", 0.7)

            # Check if function supports pag_applied_layers parameter
            import inspect

            sig = inspect.signature(self.run_function)
            if "pag_applied_layers" in sig.parameters:
                kwargs["pag_applied_layers"] = config.get("pag_applied_layers", None)


        # Call CFG model's main function
        self.run_function(**kwargs)

        print(f"CFG model has generated and saved {len(prompts)} images in real-time")


class CDGModelRunner(BaseModelRunner):
    """CDG model runner"""

    def __init__(self, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, model_variant, seed, config)
        self.run_function = self._load_run_function()

    def _load_run_function(self):
        """Load the main function of corresponding runner script"""
        # Build module path
        module_path = f"models.runner.{self.model_type}.{self.model_variant}"

        try:
            # Dynamically import module
            module = importlib.import_module(module_path)

            if not hasattr(module, "main"):
                raise AttributeError(f"Module {module_path} does not have main function")

            return module.main
        except ImportError:
            raise ImportError(f"Cannot import module: {module_path}")

    def generate_and_save(self, prompts: List[str], save_path_list: List[str]) -> None:
        """
        CDG model's generate-and-save implementation
        Uses CDG model's save_path_list parameter for true real-time saving

        Args:
            prompts: List of prompts
            save_path_list: List of save paths, should have same length as prompts
        """
        if len(prompts) != len(save_path_list):
            raise ValueError(f"Number of prompts({len(prompts)}) doesn't match number of save paths({len(save_path_list)})")

        # Ensure all save directories exist
        for save_path in save_path_list:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get configuration (prioritize external configuration)
        config = self._get_config()

        guidance_scale = config.get("guidance_scale", 7.5)
        num_inference_steps = config.get("num_inference_steps", 28)

        print(
            f"CDG model using configuration: guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}"
        )

        # Prepare basic parameters
        kwargs = {
            "prompts": prompts,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "process_params": config.get("process_params", {}),
            "seed": self.seed,
            "debug": config.get("debug", False),
            "save_path_list": save_path_list,
        }

        # Only add PAG parameters for PAG models
        if "pag" in self.model_variant.lower():
            kwargs["pag_scale"] = config.get("pag_scale", 0.7)
            kwargs["pag_applied_layers"] = config.get("pag_applied_layers", None)

        # Call CDG model's main function
        self.run_function(**kwargs)

        print(f"CDG model has generated and saved {len(prompts)} images in real-time")


class ICGModelRunner(BaseModelRunner):
    """ICG model runner"""

    def __init__(self, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, model_variant, seed, config)
        self.run_function = self._load_run_function()

    def _load_run_function(self):
        """Load the main function of corresponding runner script"""
        # Build module path
        module_path = f"models.runner.{self.model_type}.{self.model_variant}"

        try:
            # Dynamically import module
            module = importlib.import_module(module_path)

            if not hasattr(module, "main"):
                raise AttributeError(f"Module {module_path} does not have main function")

            return module.main
        except ImportError:
            raise ImportError(f"Cannot import module: {module_path}")

    def generate_and_save(self, prompts: List[str], save_path_list: List[str]) -> None:
        """
        ICG model's generate-and-save implementation
        Uses ICG model's save_path_list parameter for true real-time saving

        Args:
            prompts: List of prompts
            save_path_list: List of save paths, should have same length as prompts
        """
        if len(prompts) != len(save_path_list):
            raise ValueError(f"Number of prompts({len(prompts)}) doesn't match number of save paths({len(save_path_list)})")

        # Ensure all save directories exist
        for save_path in save_path_list:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get configuration (prioritize external configuration)
        config = self._get_config()

        guidance_scale = config.get("guidance_scale", 7.5)
        num_inference_steps = config.get("num_inference_steps", 28)
        icg_seed = config.get("icg_seed", 42)

        print(
            f"ICG model using configuration: guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}, icg_seed={icg_seed}"
        )

        # Prepare basic parameters
        kwargs = {
            "prompts": prompts,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": self.seed,
            "icg_seed": icg_seed,
            "show_random_texts": config.get("show_random_texts", False),
            "save_path_list": save_path_list,
        }

        # Call ICG model's main function
        self.run_function(**kwargs)

        print(f"ICG model has generated and saved {len(prompts)} images in real-time")


class CADSModelRunner(BaseModelRunner):
    """CADS model runner"""

    def __init__(self, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, model_variant, seed, config)
        self.run_function = self._load_run_function()

    def _load_run_function(self):
        """Load the main function of corresponding runner script"""
        # Build module path
        module_path = f"models.runner.{self.model_type}.{self.model_variant}"

        try:
            # Dynamically import module
            module = importlib.import_module(module_path)

            if not hasattr(module, "main"):
                raise AttributeError(f"Module {module_path} does not have main function")

            return module.main
        except ImportError:
            raise ImportError(f"Cannot import module: {module_path}")

    def generate_and_save(self, prompts: List[str], save_path_list: List[str]) -> None:
        """
        CADS model's generate-and-save implementation
        Uses CADS model's save_path_list parameter for true real-time saving

        Args:
            prompts: List of prompts
            save_path_list: List of save paths, should have same length as prompts
        """
        if len(prompts) != len(save_path_list):
            raise ValueError(f"Number of prompts({len(prompts)}) doesn't match number of save paths({len(save_path_list)})")

        # Ensure all save directories exist
        for save_path in save_path_list:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get configuration (prioritize external configuration)
        config = self._get_config()

        guidance_scale = config.get("guidance_scale", 7.0)
        num_inference_steps = config.get("num_inference_steps", 28)
        s = config.get("s", 0.07)  # CADS-specific noise intensity parameter

        print(
            f"CADS model using configuration: guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}, s={s}"
        )

        # Prepare parameters
        kwargs = {
            "prompts": prompts,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "s": s,
            "seed": self.seed,
            "save_path_list": save_path_list,
        }

        # Call CADS model's main function
        self.run_function(**kwargs)

        print(f"CADS model has generated and saved {len(prompts)} images in real-time")


class SEGModelRunner(BaseModelRunner):
    """SEG model runner"""

    def __init__(self, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, model_variant, seed, config)
        self.run_function = self._load_run_function()

    def _load_run_function(self):
        """Load the main function of corresponding runner script"""
        # Build module path
        module_path = f"models.runner.{self.model_type}.{self.model_variant}"

        try:
            # Dynamically import module
            module = importlib.import_module(module_path)

            if not hasattr(module, "main"):
                raise AttributeError(f"Module {module_path} does not have main function")

            return module.main
        except ImportError:
            raise ImportError(f"Cannot import module: {module_path}")

    def generate_and_save(self, prompts: List[str], save_path_list: List[str]) -> None:
        """
        SEG model's generate-and-save implementation
        Uses SEG model's save_path_list parameter for true real-time saving

        Args:
            prompts: List of prompts
            save_path_list: List of save paths, should have same length as prompts
        """
        if len(prompts) != len(save_path_list):
            raise ValueError(f"Number of prompts({len(prompts)}) doesn't match number of save paths({len(save_path_list)})")

        # Ensure all save directories exist
        for save_path in save_path_list:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get configuration (prioritize external configuration)
        config = self._get_config()

        guidance_scale = config.get("guidance_scale", 1.0)
        seg_scale = config.get("seg_scale", 0.75)
        seg_blur_sigma = config.get("seg_blur_sigma", 9999999.0)
        seg_applied_layers = config.get("seg_applied_layers", None)
        seg_applied_layers_index = config.get("seg_applied_layers_index", None)
        num_inference_steps = config.get("num_inference_steps", 28)

        print(
            f"SEG model using configuration: guidance_scale={guidance_scale}, seg_scale={seg_scale}, "
            f"seg_blur_sigma={seg_blur_sigma}, seg_applied_layers={seg_applied_layers}, "
            f"seg_applied_layers_index={seg_applied_layers_index}, num_inference_steps={num_inference_steps}"
        )

        # Prepare parameters
        kwargs = {
            "prompts": prompts,
            "guidance_scale": guidance_scale,
            "seg_scale": seg_scale,
            "seg_blur_sigma": seg_blur_sigma,
            "seg_applied_layers_index": seg_applied_layers_index,
            "num_inference_steps": num_inference_steps,
            "seed": self.seed,
            "save_path_list": save_path_list,
        }

        # Only add seg_applied_layers if the function accepts it
        sig = inspect.signature(self.run_function)
        if "seg_applied_layers" in sig.parameters:
            kwargs["seg_applied_layers"] = seg_applied_layers

        # Call SEG model's main function
        self.run_function(**kwargs)

        print(f"SEG model has generated and saved {len(prompts)} images in real-time")


class PAGModelRunner(BaseModelRunner):
    """PAG model runner"""

    def __init__(self, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, model_variant, seed, config)
        self.run_function = self._load_run_function()

    def _load_run_function(self):
        """Load the main function of corresponding runner script"""
        # Build module path (use standard pag directory structure)
        module_path = f"models.runner.{self.model_type}.{self.model_variant}"

        try:
            # Dynamically import module
            module = importlib.import_module(module_path)

            if not hasattr(module, "main"):
                raise AttributeError(f"Module {module_path} does not have main function")

            return module.main
        except ImportError:
            raise ImportError(f"Cannot import module: {module_path}")

    def generate_and_save(self, prompts: List[str], save_path_list: List[str]) -> None:
        """
        PAG model's generate-and-save implementation
        Uses PAG model's save_path_list parameter for true real-time saving

        Args:
            prompts: List of prompts
            save_path_list: List of save paths, should have same length as prompts
        """
        if len(prompts) != len(save_path_list):
            raise ValueError(f"Number of prompts({len(prompts)}) doesn't match number of save paths({len(save_path_list)})")

        # Ensure all save directories exist
        for save_path in save_path_list:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get configuration (prioritize external configuration)
        config = self._get_config()

        guidance_scale = config.get("guidance_scale", 7)
        pag_scale = config.get("pag_scale", 3)
        pag_applied_layers = config.get("pag_applied_layers", ["blocks.13"])
        num_inference_steps = config.get("num_inference_steps", 28)

        print(
            f"PAG model using configuration: guidance_scale={guidance_scale}, pag_scale={pag_scale}, "
            f"pag_applied_layers={pag_applied_layers}, num_inference_steps={num_inference_steps}"
        )

        # Prepare parameters
        kwargs = {
            "prompts": prompts,
            "guidance_scale": guidance_scale,
            "pag_scale": pag_scale,
            "pag_applied_layers": pag_applied_layers,
            "num_inference_steps": num_inference_steps,
            "seed": self.seed,
            "save_path_list": save_path_list,
        }

        # Call PAG model's main function
        self.run_function(**kwargs)

        print(f"PAG model has generated and saved {len(prompts)} images in real-time")


class DNPModelRunner(BaseModelRunner):
    """DNP (Diffusion-Negative Prompting) model runner"""

    def __init__(self, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, model_variant, seed, config)
        self.run_function = self._load_run_function()

    def _load_run_function(self):
        """Load the main function of corresponding runner script"""
        module_path = f"models.runner.{self.model_type}.{self.model_variant}"

        try:
            module = importlib.import_module(module_path)

            if not hasattr(module, "main"):
                raise AttributeError(f"Module {module_path} does not have main function")

            return module.main
        except ImportError:
            raise ImportError(f"Cannot import module: {module_path}")

    def generate_and_save(self, prompts: List[str], save_path_list: List[str]) -> None:
        """
        DNP model's generate-and-save implementation
        Uses DNP model's save_path_list parameter for true real-time saving

        Args:
            prompts: List of prompts
            save_path_list: List of save paths, should have same length as prompts
        """
        if len(prompts) != len(save_path_list):
            raise ValueError(f"Number of prompts({len(prompts)}) doesn't match number of save paths({len(save_path_list)})")

        # Ensure all save directories exist
        for save_path in save_path_list:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get configuration (prioritize external configuration)
        config = self._get_config()

        guidance_scale = config.get("guidance_scale", 7)
        num_inference_steps = config.get("num_inference_steps", 28)
        dns_guidance_scale = config.get("dns_guidance_scale", guidance_scale)
        blip2_model_path = config.get("blip2_model_path", "path/to/your/blip2-opt-2.7b")

        print(
            f"DNP model using configuration: guidance_scale={guidance_scale}, "
            f"dns_guidance_scale={dns_guidance_scale}, num_inference_steps={num_inference_steps}"
        )

        # Prepare parameters
        kwargs = {
            "prompts": prompts,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "dns_guidance_scale": dns_guidance_scale,
            "blip2_model_path": blip2_model_path,
            "seed": self.seed,
            "debug": config.get("debug", False),
            "save_path_list": save_path_list,
        }

        # Call DNP model's main function
        self.run_function(**kwargs)

        print(f"DNP model has generated and saved {len(prompts)} images in real-time")


class SFGModelRunner(BaseModelRunner):
    """SFG (Segmentation-free Guidance) model runner"""

    def __init__(self, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_type, model_variant, seed, config)
        self.run_function = self._load_run_function()

    def _load_run_function(self):
        """Load the main function of corresponding runner script"""
        module_path = f"models.runner.{self.model_type}.{self.model_variant}"

        try:
            module = importlib.import_module(module_path)

            if not hasattr(module, "main"):
                raise AttributeError(f"Module {module_path} does not have main function")

            return module.main
        except ImportError:
            raise ImportError(f"Cannot import module: {module_path}")

    def generate_and_save(self, prompts: List[str], save_path_list: List[str]) -> None:
        """
        SFG model's generate-and-save implementation
        Uses SFG model's save_path_list parameter for true real-time saving

        Args:
            prompts: List of prompts
            save_path_list: List of save paths, should have same length as prompts
        """
        if len(prompts) != len(save_path_list):
            raise ValueError(f"Number of prompts({len(prompts)}) doesn't match number of save paths({len(save_path_list)})")

        # Ensure all save directories exist
        for save_path in save_path_list:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get configuration (prioritize external configuration)
        config = self._get_config()

        guidance_scale = config.get("guidance_scale", 7)
        sfg_guidance_scale = config.get("sfg_guidance_scale", 2)
        sfg_scale = config.get("sfg_scale", 10.0)
        sfg_start_ratio = config.get("sfg_start_ratio", 0.5)
        sfg_applied_layers_index = config.get("sfg_applied_layers_index", None)
        num_inference_steps = config.get("num_inference_steps", 28)

        print(
            f"SFG model using configuration: guidance_scale={guidance_scale}, sfg_guidance_scale={sfg_guidance_scale}, "
            f"sfg_scale={sfg_scale}, sfg_start_ratio={sfg_start_ratio}, "
            f"sfg_applied_layers_index={sfg_applied_layers_index}, num_inference_steps={num_inference_steps}"
        )

        # Prepare parameters
        kwargs = {
            "prompts": prompts,
            "guidance_scale": guidance_scale,
            "sfg_guidance_scale": sfg_guidance_scale,
            "sfg_scale": sfg_scale,
            "sfg_start_ratio": sfg_start_ratio,
            "sfg_applied_layers_index": sfg_applied_layers_index,
            "num_inference_steps": num_inference_steps,
            "seed": self.seed,
            "save_path_list": save_path_list,
        }

        # Call SFG model's main function
        self.run_function(**kwargs)

        print(f"SFG model has generated and saved {len(prompts)} images in real-time")


class ModelFactory:
    """Model factory for creating corresponding model runners"""

    RUNNER_CLASSES = {
        "cfg": CFGModelRunner,
        "cdg": CDGModelRunner,
        "icg": ICGModelRunner,
        "cads": CADSModelRunner,
        "seg": SEGModelRunner,
        "pag": PAGModelRunner,
        "sfg": SFGModelRunner,
        "dnp": DNPModelRunner,
    }

    @classmethod
    def create_runner(
        cls, model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None
    ) -> BaseModelRunner:
        """
        Create model runner

        Args:
            model_type: Model type (cfg, cdg, icg, cads)
            model_variant: Model variant (sd3, flux)
            seed: Random seed for controlling randomness in generated images
            config: External configuration dictionary, use system default config if None

        Returns:
            Corresponding model runner instance
        """
        if model_type not in cls.RUNNER_CLASSES:
            raise ValueError(f"Unsupported model type: {model_type}")

        runner_class = cls.RUNNER_CLASSES[model_type]
        return runner_class(model_type, model_variant, seed, config)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported model types"""
        return list(cls.RUNNER_CLASSES.keys())


def create_model_runner(
    model_type: str, model_variant: str, seed: int = 42, config: Optional[Dict[str, Any]] = None
) -> BaseModelRunner:
    """
    Convenience function: Create model runner

    Args:
        model_type: Model type
        model_variant: Model variant
        seed: Random seed for controlling randomness in generated images
        config: External configuration dictionary

    Returns:
        Model runner instance
    """
    return ModelFactory.create_runner(model_type, model_variant, seed, config)
