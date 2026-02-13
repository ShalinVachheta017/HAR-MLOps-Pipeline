"""
=============================================================================
common.py — General-purpose file, directory, and serialization utilities
=============================================================================

Centralises repeated boilerplate that appears across almost every component:
  • directory creation      (ensure_dir)
  • YAML / JSON read-write  (read_yaml, write_yaml, read_json, write_json)
  • NumPy I/O               (load_numpy, save_numpy)
  • timestamp generation    (get_timestamp)
  • file validation         (validate_file_exists, get_file_size)

Usage:
    from src.utils.common import ensure_dir, read_yaml, get_timestamp
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ─── Directory helpers ────────────────────────────────────────────────

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if they don't exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─── YAML I/O ────────────────────────────────────────────────────────

def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    import yaml  # lazy import — yaml may not be needed everywhere

    path = Path(path)
    validate_file_exists(path, "YAML config")
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    logger.debug("Loaded YAML: %s (%d keys)", path.name, len(content))
    return content


def write_yaml(path: Union[str, Path], data: Dict[str, Any]) -> Path:
    """Write a dictionary to a YAML file.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    data : dict
        Data to serialize.

    Returns
    -------
    Path
        The written file path.
    """
    import yaml

    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.debug("Wrote YAML: %s", path.name)
    return path


# ─── JSON I/O ────────────────────────────────────────────────────────

def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a JSON file and return its contents.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    dict or list
        Parsed JSON content.
    """
    path = Path(path)
    validate_file_exists(path, "JSON file")
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    logger.debug("Loaded JSON: %s", path.name)
    return content


def write_json(
    path: Union[str, Path],
    data: Any,
    indent: int = 2,
) -> Path:
    """Write data to a JSON file.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    data : Any
        JSON-serialisable data.
    indent : int
        Pretty-print indentation (default 2).

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug("Wrote JSON: %s", path.name)
    return path


# ─── NumPy I/O ───────────────────────────────────────────────────────

def load_numpy(path: Union[str, Path]) -> np.ndarray:
    """Load a `.npy` file.

    Parameters
    ----------
    path : str or Path
        Path to the `.npy` file.

    Returns
    -------
    np.ndarray
    """
    path = Path(path)
    validate_file_exists(path, "NumPy array")
    arr = np.load(path, allow_pickle=False)
    logger.debug("Loaded NumPy: %s  shape=%s  dtype=%s", path.name, arr.shape, arr.dtype)
    return arr


def save_numpy(path: Union[str, Path], arr: np.ndarray) -> Path:
    """Save a NumPy array to a `.npy` file.

    Parameters
    ----------
    path : str or Path
        Destination file path.
    arr : np.ndarray
        Array to save.

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)
    logger.debug("Saved NumPy: %s  shape=%s  dtype=%s", path.name, arr.shape, arr.dtype)
    return path


# ─── Timestamp ───────────────────────────────────────────────────────

def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return current UTC-naive timestamp string.

    Parameters
    ----------
    fmt : str
        strftime format (default ``'%Y%m%d_%H%M%S'``).

    Returns
    -------
    str
        Formatted timestamp, e.g. ``'20260213_143022'``.
    """
    return datetime.now().strftime(fmt)


# ─── File helpers ────────────────────────────────────────────────────

def validate_file_exists(path: Union[str, Path], description: str = "File") -> None:
    """Raise ``FileNotFoundError`` if *path* does not exist.

    Parameters
    ----------
    path : str or Path
        File path to check.
    description : str
        Human-readable description for the error message.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def get_file_size(path: Union[str, Path]) -> str:
    """Return human-readable file size string.

    Parameters
    ----------
    path : str or Path
        File whose size to report.

    Returns
    -------
    str
        e.g. ``'12.4 MB'``, ``'3.1 KB'``.
    """
    path = Path(path)
    if not path.exists():
        return "N/A"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
