#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# Set project path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Delayed import to avoid dependency issues
def _import_modules():
    """Delayed import of submodules"""
    try:
        from . import utils
        from . import metrics
        from . import core

        return utils, metrics, core
    except ImportError as e:
        # If there are dependency issues, only import modules that don't depend on torch
        from . import utils
        from . import core

        metrics = None
        print(f"Warning: Unable to import metrics module: {e}")
        return utils, metrics, core


__all__ = []


# Provide convenient access interfaces
def get_utils():
    """Get utils module"""
    from . import utils

    return utils


def get_core():
    """Get core module"""
    from . import core

    return core


def get_metrics():
    """Get metrics module"""
    try:
        from . import metrics

        return metrics
    except ImportError as e:
        print(f"Warning: Unable to import metrics module: {e}")
        return None
