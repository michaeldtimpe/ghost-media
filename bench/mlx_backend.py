#!/usr/bin/env python3
"""
Thin re-export. The MLX vision backend is defined in `vision_backends.py` so the
`get_backend` factory stays the single authority (and `enrich_analyses.py
--backend mlx` works for the pilot rescan with no extra wiring). This module
exists only for discoverability from within the bench package.
"""

from vision_backends import MLXVisionBackend  # noqa: F401

__all__ = ["MLXVisionBackend"]
