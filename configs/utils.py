#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigManager:
    """Configuration manager for unified management of all configuration file reading"""
    
    def __init__(self):
        # Get configuration file root directory
        self.config_root = Path(__file__).parent
        self.models_config_path = self.config_root / "models" / "models_path.json"
        self.methods_config_dir = self.config_root / "methods"
        self.evaluations_config_dir = self.config_root / "evaluations"
        
        # Cache configuration content
        self._models_config_cache = None
        self._methods_config_cache = {}
        self._evaluations_config_cache = {}
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get model path
        
        Args:
            model_name: Model name, such as 'sd3', 'flux'
            
        Returns:
            Model path
            
        Raises:
            ValueError: If model name does not exist
        """
        if self._models_config_cache is None:
            with open(self.models_config_path, 'r', encoding='utf-8') as f:
                try:
                    self._models_config_cache = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in {self.models_config_path}: {e}")
        
        if model_name not in self._models_config_cache:
            available_models = list(self._models_config_cache.keys())
            raise ValueError(f"Model '{model_name}' does not exist. Available models: {available_models}")
        
        return self._models_config_cache[model_name]
    
    def get_method_config(self, method_name: str, model_name: str) -> Dict[str, Any]:
        """
        Get method configuration
        
        Args:
            method_name: Method name, such as 'cdg', 'cfg'
            model_name: Model name, such as 'sd3', 'flux'
            
        Returns:
            Method configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file does not exist
        """
        config_key = f"{method_name}_{model_name}"
        
        if config_key not in self._methods_config_cache:
            config_file = self.methods_config_dir / f"{config_key}.json"
            if not config_file.exists():
                raise FileNotFoundError(f"Method configuration file does not exist: {config_file}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                try:
                    self._methods_config_cache[config_key] = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in {config_file}: {e}")
        
        return self._methods_config_cache[config_key]
    
    def get_evaluation_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get evaluation configuration
        
        Args:
            config_name: Configuration name, such as 'coco2017', 'metrics'
            
        Returns:
            Evaluation configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file does not exist
        """
        if config_name not in self._evaluations_config_cache:
            config_file = self.evaluations_config_dir / f"{config_name}.json"
            if not config_file.exists():
                raise ValueError(f"Evaluation configuration file does not exist: {config_file}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                try:
                    self._evaluations_config_cache[config_name] = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in {config_file}: {e}")
        
        return self._evaluations_config_cache[config_name]
    
    def get_available_models(self) -> list:
        """Get list of all available models"""
        if self._models_config_cache is None:
            with open(self.models_config_path, 'r', encoding='utf-8') as f:
                try:
                    self._models_config_cache = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in {self.models_config_path}: {e}")
        return list(self._models_config_cache.keys())
    
    def get_available_methods(self) -> Dict[str, list]:
        """Get list of all available methods"""
        methods = {}
        for config_file in self.methods_config_dir.glob("*.json"):
            # Parse filename, format is method_model.json
            name_parts = config_file.stem.split('_', 1)
            if len(name_parts) == 2:
                method_name, model_name = name_parts
                if method_name not in methods:
                    methods[method_name] = []
                methods[method_name].append(model_name)
        return methods
    
    def get_available_evaluations(self) -> list:
        """Get list of all available evaluation configurations"""
        return [config_file.stem for config_file in self.evaluations_config_dir.glob("*.json")]
    
    def reload_cache(self):
        """Reload all caches"""
        self._models_config_cache = None
        self._methods_config_cache = {}
        self._evaluations_config_cache = {}


# Create global configuration manager instance
config_manager = ConfigManager()


# Convenience functions
def get_model_path(model_name: str) -> str:
    """Convenience function to get model path"""
    return config_manager.get_model_path(model_name)


def get_method_config(method_name: str, model_name: str) -> Dict[str, Any]:
    """Convenience function to get method configuration"""
    return config_manager.get_method_config(method_name, model_name)


def get_evaluation_config(config_name: str) -> Dict[str, Any]:
    """Convenience function to get evaluation configuration"""
    return config_manager.get_evaluation_config(config_name)


def get_available_models() -> list:
    """Convenience function to get all available models"""
    return config_manager.get_available_models()


def get_available_methods() -> Dict[str, list]:
    """Convenience function to get all available methods"""
    return config_manager.get_available_methods()


def get_available_evaluations() -> list:
    """Convenience function to get all available evaluation configurations"""
    return config_manager.get_available_evaluations()


def validate_model_method_combination(method_name: str, model_name: str) -> bool:
    """
    Validate whether the combination of method and model is valid
    
    Args:
        method_name: Method name
        model_name: Model name
        
    Returns:
        True if combination is valid, False otherwise
    """
    try:
        get_method_config(method_name, model_name)
        get_model_path(model_name)
        return True
    except (FileNotFoundError, ValueError):
        return False


def parse_model_identifier(model_identifier: str) -> tuple:
    """
    Parse model identifier
    
    Args:
        model_identifier: String in format 'method/model' or 'method_model'
        
    Returns:
        (method_name, model_name) tuple
        
    Raises:
        ValueError: If format is incorrect
    """
    if '/' in model_identifier:
        parts = model_identifier.split('/', 1)
    elif '_' in model_identifier:
        parts = model_identifier.split('_', 1)
    else:
        raise ValueError(f"Invalid model identifier format: {model_identifier}, should be 'method/model' or 'method_model'")
    
    if len(parts) != 2:
        raise ValueError(f"Invalid model identifier format: {model_identifier}, should be 'method/model' or 'method_model'")
    
    method_name, model_name = parts
    return method_name.strip(), model_name.strip()


class ConfigurationManager:
    """Unified configuration manager"""
    
    @staticmethod
    def load_and_validate_config(config_path: str, method_name: str, model_name: str) -> Dict[str, Any]:
        """
        Enhanced configuration loading and validation function, supporting multiple path formats and better error handling
        
        Args:
            config_path: Configuration file path
            method_name: Method name
            model_name: Model name
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file does not exist and cannot be obtained from configs
            ValueError: If configuration validation fails
        """
        config_data = None
        config_source = None
        
        # 1. First try to load specified path directly
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                try:
                    config_data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in {config_path}: {e}")
            config_source = f"Specified path: {config_path}"
            
        # 2. If direct path does not exist, try loading from configs/methods/ directory
        elif not config_data:
            # Support multiple naming formats
            possible_names = [
                f"{method_name}_{model_name}.json",
                f"{method_name}+pag_{model_name}.json",  # Support PAG variants
                os.path.basename(config_path)  # Use filename directly
            ]
            
            for name in possible_names:
                try:
                    # Try using standard method_model format
                    base_name = name.replace('.json', '')
                    if '+' in base_name:
                        # Handle PAG variant format
                        parts = base_name.split('+')
                        if len(parts) == 2:
                            method_part = parts[0]
                            variant_model = parts[1].split('_')
                            if len(variant_model) == 2:
                                variant, model = variant_model
                                if variant == 'pag':
                                    config_data = get_method_config(f"{method_part}+{variant}", model)
                                    config_source = f"configs default configuration: {method_part}+{variant}_{model}"
                                    break
                    else:
                        # Standard format
                        parts = base_name.split('_', 1)
                        if len(parts) == 2:
                            method, model = parts
                            config_data = get_method_config(method, model)
                            config_source = f"configs default configuration: {method}_{model}"
                            break
                except FileNotFoundError:
                    continue
            
            # 3. If still not found, try standard method_name and model_name combination
            if not config_data:
                try:
                    config_data = get_method_config(method_name, model_name)
                    config_source = f"configs default configuration: {method_name}_{model_name}"
                except FileNotFoundError:
                    pass
        
        # 4. If all methods fail, provide detailed error information
        if not config_data:
            available_methods = get_available_methods()
            suggestion = f"Available configuration combinations:\n"
            for method, models in available_methods.items():
                suggestion += f"  {method}: {models}\n"
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}\n{suggestion}")
        
        
        print(f"Loaded configuration from {config_source}")
        print(f"Configuration content: guidance_scale={config_data.get('guidance_scale', 'N/A')}, num_inference_steps={config_data.get('num_inference_steps', 'N/A')}")
        
        return config_data
    
    @staticmethod
    def _validate_config(config: Dict[str, Any], method_name: str, model_name: str) -> bool:
        """Enhanced configuration validation function providing more comprehensive validation logic"""
        try:
            # Basic parameter validation
            if 'guidance_scale' in config:
                if not isinstance(config['guidance_scale'], (int, float)) or config['guidance_scale'] <= 0:
                    print(f"Configuration validation failed: guidance_scale must be a positive number, current value: {config['guidance_scale']}")
                    return False
            
            if 'num_inference_steps' in config:
                if not isinstance(config['num_inference_steps'], int) or config['num_inference_steps'] <= 0:
                    print(f"Configuration validation failed: num_inference_steps must be a positive integer, current value: {config['num_inference_steps']}")
                    return False
            
            if 'seed' in config:
                if not isinstance(config['seed'], int):
                    print(f"Configuration validation failed: seed must be an integer, current value: {config['seed']}")
                    return False
            
            # Model-specific validation
            if model_name in ['flux']:
                flux_fields = ['height', 'width', 'max_sequence_length']
                for field in flux_fields:
                    if field not in config:
                        print(f"Warning: Flux model configuration missing recommended field: {field}")
                    elif field in ['height', 'width'] and not isinstance(config[field], int):
                        print(f"Configuration validation failed: {field} must be an integer, current value: {config[field]}")
                        return False
            
            # Method-specific validation
            if method_name == 'cdg' or method_name.startswith('cdg'):
                if 'process_params' not in config:
                    print(f"Configuration validation failed: CDG method missing process_params field")
                    return False
                
                process_params = config['process_params']
                required_keys = ["process_index", "keep_ratio", "separate_clip_t5", "calculate_params"]
                missing_keys = [key for key in required_keys if key not in process_params]
                if missing_keys:
                    print(f"Configuration validation failed: process_params missing fields: {missing_keys}")
                    return False
            
            # PAG-specific validation
            if '+pag' in method_name or 'pag' in method_name.lower():
                if 'pag_scale' not in config:
                    print(f"Warning: PAG method configuration missing pag_scale field")
                elif not isinstance(config['pag_scale'], (int, float)):
                    print(f"Configuration validation failed: pag_scale must be a number, current value: {config['pag_scale']}")
                    return False
                
                if 'pag_applied_layers' not in config:
                    print(f"Warning: PAG method configuration missing pag_applied_layers field")
                elif not isinstance(config['pag_applied_layers'], list):
                    print(f"Configuration validation failed: pag_applied_layers must be a list, current value: {config['pag_applied_layers']}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error occurred during configuration validation: {e}")
            return False
    
    @staticmethod
    def prepare_model_config(config: Dict[str, Any], additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare configuration parameters to be passed to the model
        
        Args:
            config: Configuration dictionary
            additional_params: Additional parameters
            
        Returns:
            Prepared configuration dictionary
        """
        prepared_config = config.copy()
        
        if additional_params:
            prepared_config.update(additional_params)
        
        return prepared_config


class ParameterManager:
    """Unified parameter manager"""
    
    @staticmethod
    def get_evaluation_parameters(config_type: str = "coco2017") -> Dict[str, Any]:
        """
        Get evaluation parameters, prioritize reading from configs, if failed then raise error
        
        Args:
            config_type: Configuration type
            
        Returns:
            Evaluation parameters dictionary
            
        Raises:
            FileNotFoundError: If unable to obtain configuration from configs
            ValueError: If configuration validation fails
        """
        try:
            # Get dataset configuration
            dataset_config = get_evaluation_config(config_type)
            gt_path = dataset_config.get("image_path")
            
            # Get metrics configuration
            metrics_config = get_evaluation_config("metrics")
            
            # Validate path
            if not gt_path or not os.path.exists(gt_path):
                raise ValueError(f"Ground truth image path does not exist or is not configured: {gt_path}")
            
            print(f"Loading evaluation settings from configs:")
            print(f"  Ground truth image path: {gt_path}")
            print(f"  Metrics configuration: {list(metrics_config.keys())}")
            
            return {
                "gt_path": gt_path,
                "metrics_config": metrics_config,
                "dataset_config": dataset_config
            }
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Unable to obtain evaluation configuration from configs: {e}")
        except Exception as e:
            raise ValueError(f"Evaluation configuration validation failed: {e}")
    
    @staticmethod
    def prepare_metric_kwargs(metric_name: str, config: Dict[str, Any], batch_size: int = None) -> Dict[str, Any]:
        """
        Prepare metric parameters, prioritize using configuration from configs
        
        Args:
            metric_name: Metric name
            config: Metric configuration
            batch_size: Batch size (optional, overrides value in configuration)
            
        Returns:
            Prepared metric parameters
            
        Raises:
            ValueError: If metric configuration is incomplete
        """
        if not config:
            raise ValueError(f"Configuration for metric {metric_name} is empty")
        
        # Basic parameters
        metric_kwargs = {
            'device': config.get('device', 'cuda'),
            'batch_size': batch_size or config.get('batch_size', 250)
        }
        
        # Add specific parameters based on metric type
        if metric_name == 'CLIPScore':
            if 'name' not in config:
                raise ValueError(f"CLIPScore configuration missing 'name' field")
            metric_kwargs['model_name'] = config['name']
            
        elif metric_name == 'FID':
            metric_kwargs['feature_dim'] = config.get('feature_dim', 2048)
            
        elif metric_name == 'VQAScore':
            required_fields = ['name', 'question_template', 'answer_template']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"VQAScore configuration missing '{field}' field")
            
            metric_kwargs.update({
                'model_name': config['name'],
                'question_template': config['question_template'],
                'answer_template': config['answer_template']
            })
            
        elif metric_name == 'AestheticScore':
            # AestheticScore needs encoder_model and predictor_path for local models
            if 'encoder_model' in config:
                metric_kwargs['encoder_model'] = config['encoder_model']
            if 'predictor_path' in config:
                metric_kwargs['predictor_path'] = config['predictor_path']
            
        elif metric_name == 'HPScoreV2':
            # HPScoreV2 usually only needs basic parameters
            pass
            
        else:
            print(f"Warning: Unknown metric type {metric_name}, using basic parameters")
        
        return metric_kwargs
    
    @staticmethod
    def get_prompt_source_config() -> Dict[str, Any]:
        """
        Get prompt source configuration
        
        Returns:
            Prompt source configuration dictionary
            
        Raises:
            FileNotFoundError: If unable to obtain COCO configuration
        """
        try:
            coco_config = get_evaluation_config("coco2017")
            return {
                "coco_annotation_file": coco_config.get("prompt_file_path"),
                "coco_image_path": coco_config.get("image_path")
            }
        except FileNotFoundError:
            raise FileNotFoundError("Unable to obtain COCO configuration from configs, please check configs/evaluations/coco2017.json")


# Add convenience functions
def load_and_validate_config(config_path: str, method_name: str, model_name: str) -> Dict[str, Any]:
    """Convenience function to load and validate configuration"""
    return ConfigurationManager.load_and_validate_config(config_path, method_name, model_name)


def prepare_model_config(config: Dict[str, Any], additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to prepare model configuration"""
    return ConfigurationManager.prepare_model_config(config, additional_params)


def get_evaluation_parameters(config_type: str = "coco2017") -> Dict[str, Any]:
    """Convenience function to get evaluation parameters"""
    return ParameterManager.get_evaluation_parameters(config_type)


def prepare_metric_kwargs(metric_name: str, config: Dict[str, Any], batch_size: int = None) -> Dict[str, Any]:
    """Convenience function to prepare metric parameters"""
    return ParameterManager.prepare_metric_kwargs(metric_name, config, batch_size)


def get_prompt_source_config() -> Dict[str, Any]:
    """Convenience function to get prompt source configuration"""
    return ParameterManager.get_prompt_source_config()


if __name__ == "__main__":
    # Test code
    print("Available models:", get_available_models())
    print("Available methods:", get_available_methods())
    print("Available evaluation configurations:", get_available_evaluations())
    
    # Test model path retrieval
    try:
        print("SD3 model path:", get_model_path("sd3"))
    except Exception as e:
        print("Failed to get SD3 model path:", e)
    
    # Test method configuration retrieval
    try:
        print("CDG SD3 configuration:", get_method_config("cdg", "sd3"))
    except Exception as e:
        print("Failed to get CDG SD3 configuration:", e)
