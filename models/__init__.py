"""
Unified text-to-image generation model interface
"""

# Version information
__version__ = "1.0.0"

# Import main modules
from . import runner
from . import pipelines

# Export all modules
__all__ = ["runner", "pipelines", "__version__"]
