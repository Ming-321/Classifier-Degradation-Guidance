import sys
from pathlib import Path

# Add project root to sys.path for importing configuration management tools
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.utils import get_method_config


def _needs_cross_attention_for_importance(degrade_ratio: dict) -> bool:
    """
    Check if cross-attention is needed for importance calculation.
    
    When degrade_ratio is {"content": 1, "padding": 0}, all content tokens are fully degraded
    without needing importance calculation, so cross-attention is not required.
    
    Args:
        degrade_ratio: Dictionary with degradation ratios
        
    Returns:
        bool: True if cross-attention is needed, False otherwise
    """
    # Special case: full content degradation without importance calculation
    if (degrade_ratio.get("content") == 1 and 
        degrade_ratio.get("padding") == 0 and
        len(degrade_ratio) == 2):
        return False
    return True


def get_sd3_cdg_default_process_params():
    """
    Get default processing parameters for SD3 CDG.

    Returns:
        dict: Dictionary containing default CDG processing parameters with the following structure:
            - process_index (int): Index of the processing step
            - keep_ratio (dict): Ratios for keeping content/padding tokens
            - separate_clip_t5 (bool): Whether to separate CLIP and T5 tokens
            - calculate_params (dict): Parameters for importance calculation algorithms
    """
    try:
        config = get_method_config("cdg", "sd3")
        return config.get("process_params", {})
    except Exception as e:
        print(
            f"Warning: Could not read SD3 CDG parameters from config file, using default values: {e}"
        )
    


def get_flux_cdg_default_process_params():
    """
    Get default processing parameters for Flux CDG.

    Returns:
        dict: Dictionary containing default CDG processing parameters with the following structure:
            - process_index (int): Index of the processing step
            - keep_ratio (dict): Ratios for keeping content/padding tokens
            - separate_clip_t5 (bool): Whether to separate CLIP and T5 tokens
            - calculate_params (dict): Parameters for importance calculation algorithms
    """
    try:
        config = get_method_config("cdg", "flux")
        return config.get("process_params", {})
    except Exception as e:
        print(
            f"Warning: Could not read Flux CDG parameters from config file, using default values: {e}"
        )


def deep_merge_dict(default_dict, update_dict):
    """
    Recursively merge two dictionaries, with values from update_dict overriding default_dict.

    Args:
        default_dict (dict): The base dictionary with default values
        update_dict (dict): The dictionary containing updates to apply

    Returns:
        dict: A new dictionary with merged values from both input dictionaries

    Note:
        If both values for a key are dictionaries, they are recursively merged.
        Otherwise, the value from update_dict overwrites the value from default_dict.
    """
    import copy

    result = copy.deepcopy(default_dict)

    for key, value in update_dict.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                # If both values are dictionaries, merge them recursively
                result[key] = deep_merge_dict(result[key], value)
            else:
                # Otherwise, directly overwrite
                result[key] = value
        else:
            # If key doesn't exist in default dictionary, add it directly
            result[key] = value

    return result


def update_sd3_cdg_process_params(process_params):
    """
    Update SD3 CDG processing parameters with support for partial updates of nested dictionaries.

    Args:
        process_params (dict): Parameters dictionary to update. Must contain "process_index" and "degrade_ratio".

    Returns:
        dict: Complete parameters dictionary with updated values merged with defaults

    Raises:
        ValueError: If required parameters are missing or have padding values
        TypeError: If parameters have incorrect types

    Examples:
        # Update with degrade_ratio that gets converted to keep_ratio
        update_params = {
            "process_index": 5,
            "degrade_ratio": {
                "content": 0.3,
                "padding": 0.7
            },
            "calculate_params": {
                "epsilon": 1e-5,
                "max_iterations": 30
            }
        }
        result = update_sd3_cdg_process_params(update_params)
        # result["keep_ratio"]["content"] becomes 0.7 (1 - 0.3)
        # result["keep_ratio"]["padding"] becomes 0.3 (1 - 0.7)
    """
    # Step 1: Get default parameters
    default_process_params = get_sd3_cdg_default_process_params()

    if process_params is None:
        raise ValueError("process_params cannot be None. Must contain 'process_index' and 'degrade_ratio'.")
    
    if not isinstance(process_params, dict):
        raise TypeError("process_params must be a dictionary")

    # Step 2: Validate required parameters
    if "process_index" not in process_params:
        raise ValueError("Missing required parameter 'process_index'")
    
    if "degrade_ratio" not in process_params:
        raise ValueError("Missing required parameter 'degrade_ratio'")
    
    # Validate process_index
    process_index = process_params["process_index"]
    if not isinstance(process_index, int):
        raise TypeError("'process_index' must be an integer")
    
    if not (0 <= process_index <= 27):
        raise ValueError(f"'process_index' must be between 0 and 27, got {process_index}")
    
    # Validate degrade_ratio
    degrade_ratio = process_params["degrade_ratio"]
    if not isinstance(degrade_ratio, dict):
        raise TypeError("'degrade_ratio' must be a dictionary")
    
    if not degrade_ratio:
        raise ValueError("'degrade_ratio' cannot be empty")
    
    valid_keys = {"content", "padding", "all"}
    if not set(degrade_ratio.keys()).issubset(valid_keys):
        raise ValueError(f"'degrade_ratio' can only contain keys {valid_keys}, got {set(degrade_ratio.keys())}")
    
    for key, value in degrade_ratio.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"'degrade_ratio[{key}]' must be a number, got {type(value).__name__}")
        if not (0 <= value <= 1):
            raise ValueError(f"'degrade_ratio[{key}]' must be between 0 and 1, got {value}")

    # Step 3: Convert degrade_ratio to keep_ratio
    import copy
    process_params_copy = copy.deepcopy(process_params)
    
    keep_ratio = {}
    for key, value in degrade_ratio.items():
        keep_ratio[key] = 1 - value
    
    # Replace degrade_ratio with keep_ratio
    process_params_copy["keep_ratio"] = keep_ratio
    del process_params_copy["degrade_ratio"]

    # Step 4: Deep merge with defaults
    return deep_merge_dict(default_process_params, process_params_copy)


def update_flux_cdg_process_params(process_params):
    """
    Update Flux CDG processing parameters with support for partial updates of nested dictionaries.

    Args:
        process_params (dict): Parameters dictionary to update. Must contain "process_index" and "degrade_ratio".

    Returns:
        dict: Complete parameters dictionary with updated values merged with defaults

    Raises:
        ValueError: If required parameters are missing or have padding values
        TypeError: If parameters have incorrect types

    Examples:
        # Update with degrade_ratio that gets converted to keep_ratio
        update_params = {
            "process_index": 5,
            "degrade_ratio": {
                "content": 0.3,
                "padding": 0.7
            },
            "calculate_params": {
                "epsilon": 1e-5,
                "max_iterations": 30
            }
        }
        result = update_flux_cdg_process_params(update_params)
        # result["keep_ratio"]["content"] becomes 0.7 (1 - 0.3)
        # result["keep_ratio"]["padding"] becomes 0.3 (1 - 0.7)
    """
    # Step 1: Get default parameters
    default_process_params = get_flux_cdg_default_process_params()

    if process_params is None:
        raise ValueError("process_params cannot be None. Must contain 'process_index' and 'degrade_ratio'.")
    
    if not isinstance(process_params, dict):
        raise TypeError("process_params must be a dictionary")

    # Step 2: Validate required parameters
    if "process_index" not in process_params:
        raise ValueError("Missing required parameter 'process_index'")
    
    if "degrade_ratio" not in process_params:
        raise ValueError("Missing required parameter 'degrade_ratio'")
    
    # Validate process_index
    process_index = process_params["process_index"]
    if not isinstance(process_index, int):
        raise TypeError("'process_index' must be an integer")
    
    if not (0 <= process_index <= 27):
        raise ValueError(f"'process_index' must be between 0 and 27, got {process_index}")
    
    # Validate degrade_ratio
    degrade_ratio = process_params["degrade_ratio"]
    if not isinstance(degrade_ratio, dict):
        raise TypeError("'degrade_ratio' must be a dictionary")
    
    if not degrade_ratio:
        raise ValueError("'degrade_ratio' cannot be empty")
    
    valid_keys = {"content", "padding", "all"}
    if not set(degrade_ratio.keys()).issubset(valid_keys):
        raise ValueError(f"'degrade_ratio' can only contain keys {valid_keys}, got {set(degrade_ratio.keys())}")
    
    for key, value in degrade_ratio.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"'degrade_ratio[{key}]' must be a number, got {type(value).__name__}")
        if not (0 <= value <= 1):
            raise ValueError(f"'degrade_ratio[{key}]' must be between 0 and 1, got {value}")

    # Step 3: Convert degrade_ratio to keep_ratio
    import copy
    process_params_copy = copy.deepcopy(process_params)
    
    keep_ratio = {}
    for key, value in degrade_ratio.items():
        keep_ratio[key] = 1 - value
    
    # Replace degrade_ratio with keep_ratio
    process_params_copy["keep_ratio"] = keep_ratio
    del process_params_copy["degrade_ratio"]

    # Step 4: Deep merge with defaults
    return deep_merge_dict(default_process_params, process_params_copy)
