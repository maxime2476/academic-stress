"""
Centralised configuration loader.

All modules should call ``get_config()`` to access parameters, never reading
config.yaml directly.  This ensures a single parse per process and makes
parameter injection easy in tests (override ``_CONFIG``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of this file's parent (src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CONFIG: dict[str, Any] | None = None


def get_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return parsed config dict, loading it on the first call.

    Parameters
    ----------
    path:
        Optional explicit path to ``config.yaml``.  When *None* the default
        ``<project_root>/config.yaml`` is used.  Useful for tests that supply
        a temporary config file.
    """
    global _CONFIG
    if _CONFIG is None or path is not None:
        config_path = Path(path) if path else _PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {config_path}. "
                "Run from the project root or pass an explicit path."
            )
        with open(config_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if path is None:
            _CONFIG = loaded
        else:
            return loaded
    return _CONFIG


def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return _PROJECT_ROOT


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return _PROJECT_ROOT / relative


def reset_config() -> None:
    """Clear cached config (used in tests)."""
    global _CONFIG
    _CONFIG = None
