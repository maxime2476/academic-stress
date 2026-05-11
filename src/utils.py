"""
Shared utilities: logging setup, JSON serialisation helpers, path management.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure root logger with console + optional file handler.

    Call once at the top of each script entry-point.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (including parents) if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to native Python types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialise *data* to JSON at *path* using ``NumpyEncoder``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, cls=NumpyEncoder, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def print_section(title: str, width: int = 70) -> None:
    """Print a section header to stdout."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")
