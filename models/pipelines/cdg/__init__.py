"""
CDG (Context-aware Weight Generation) Pipeline Package
"""

# Import utility functions
from .utils import (
    get_sd3_cdg_default_process_params,
    get_flux_cdg_default_process_params,
    update_sd3_cdg_process_params,
    update_flux_cdg_process_params,
)

from .process_token import ProcessToken
from .calculate_importance import CalculateImportance

# Import specific pipeline implementations
try:
    from .flux.pipeline import CDGFluxPipeline
except ImportError:
    CDGFluxPipeline = None

try:
    from .sd3.pipeline import CDGSD3Pipeline
except ImportError:
    CDGSD3Pipeline = None

try:
    from .qwenimage.pipeline import CDGQwenImagePipeline
except ImportError:
    CDGQwenImagePipeline = None


__all__ = [
    # Utility functions
    "get_sd3_cdg_default_process_params",
    "get_flux_cdg_default_process_params",
    "update_sd3_cdg_process_params",
    "update_flux_cdg_process_params",
    # Core classes
    "ProcessToken",
    "CalculateImportance",
    # Pipeline classes
    "CDGFluxPipeline",
    "CDGSD3Pipeline",
    "CDGQwenImagePipeline",
]
