#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib


def sanitize_filename(prompt: str, max_length: int = 50) -> str:
    """
    Convert prompt to safe filename

    Args:
        prompt: Prompt
        max_length: Maximum length

    Returns:
        str: Safe filename
    """
    # Remove unsafe characters
    safe_name = "".join([c if c.isalnum() or c in " _-" else "_" for c in prompt])
    # Replace spaces
    safe_name = safe_name.replace(" ", "_")
    # Limit length
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]

    # Add hash to ensure uniqueness
    prompt_hash = hashlib.sha1(prompt.encode()).hexdigest()[:8]

    return f"{safe_name}_{prompt_hash}"
