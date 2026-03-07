#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path


def setup_project_path():
    """Set project path to ensure proper module imports"""
    # Get project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent

    # Add to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def check_dependencies():
    """Check dependencies and environment"""
    try:
        import torch
        import transformers
        import diffusers

        print("✓ Core dependency check passed")
        return True
    except ImportError as e:
        print(f"✗ Dependency check failed: {e}")
        return False


def check_model_paths():
    """Check if model paths are correct"""
    try:
        from ..core.paths import paths

        models_dir = paths.models_root
        if not models_dir.exists():
            print(f"✗ Model directory does not exist: {models_dir}")
            return False

        print(f"✓ Model directory exists: {models_dir}")
        return True
    except ImportError:
        print("✗ Cannot import path management module")
        return False


def check_project_structure():
    """Check if project structure is correct"""
    try:
        from ..core.paths import paths

        required_dirs = [paths.evaluation_root, paths.models_root, paths.outputs_root, paths.configs_root]

        all_exist = True
        for dir_path in required_dirs:
            if dir_path.exists():
                print(f"✓ {dir_path.name} directory exists")
            else:
                print(f"✗ {dir_path.name} directory does not exist: {dir_path}")
                all_exist = False

        return all_exist
    except ImportError:
        print("✗ Cannot import path management module")
        return False


def check_environment():
    """Comprehensive environment check"""
    print("Checking environment...")

    # Set project path
    project_root = setup_project_path()
    print(f"✓ Project root directory: {project_root}")

    # Check dependencies
    deps_ok = check_dependencies()

    # Check project structure
    struct_ok = check_project_structure()

    # Check model paths
    model_ok = check_model_paths()

    if deps_ok and struct_ok and model_ok:
        print("\n✓ Environment check passed")
        return True
    else:
        print("\n✗ Environment check failed")
        return False
