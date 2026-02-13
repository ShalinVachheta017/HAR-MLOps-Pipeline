"""
=============================================================================
main_utils.py — ML / pipeline-specific utility functions
=============================================================================

Provides domain-aware helpers for the HAR MLOps pipeline:
  • Keras model loading / saving with error handling
  • Prediction CSV loading (standardised format)
  • Pipeline configuration from YAML
  • Sensor & activity label accessors
  • Class distribution computation (shared by evaluation + monitoring)
  • Per-stage logger setup
  • File archiving

Usage:
    from src.utils.main_utils import load_model, load_predictions
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.common import (
    ensure_dir,
    read_json,
    read_yaml,
    validate_file_exists,
    get_timestamp,
)

logger = logging.getLogger(__name__)

# ─── Constants (mirrors src/config.py for standalone use) ─────────────

_DEFAULT_ACTIVITY_LABELS: List[str] = [
    "ear_rubbing",
    "forehead_rubbing",
    "hair_pulling",
    "hand_scratching",
    "hand_tapping",
    "knuckles_cracking",
    "nail_biting",
    "nape_rubbing",
    "sitting",
    "smoking",
    "standing",
]

_DEFAULT_SENSOR_COLUMNS: List[str] = [
    "Ax_w", "Ay_w", "Az_w",
    "Gx_w", "Gy_w", "Gz_w",
]


# ─── Activity / sensor accessors ─────────────────────────────────────

def get_activity_labels() -> List[str]:
    """Return the canonical list of 11 HAR activity labels."""
    return list(_DEFAULT_ACTIVITY_LABELS)


def get_sensor_columns() -> List[str]:
    """Return the 6 sensor-axis column names used throughout the pipeline."""
    return list(_DEFAULT_SENSOR_COLUMNS)


# ─── Model I/O ───────────────────────────────────────────────────────

def load_model(path: Union[str, Path]) -> Any:
    """Load a Keras ``.keras`` (or ``.h5``) model with error handling.

    Parameters
    ----------
    path : str or Path
        Path to the saved model file.

    Returns
    -------
    keras.Model
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    RuntimeError
        If TensorFlow / Keras cannot load the model.
    """
    path = Path(path)
    validate_file_exists(path, "Model file")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(path))
        logger.info("Loaded model: %s  (params=%s)", path.name,
                     f"{model.count_params():,}")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model {path}: {exc}") from exc


def save_model(model: Any, path: Union[str, Path]) -> Path:
    """Save a Keras model to *path* (creates parent dirs).

    Parameters
    ----------
    model : keras.Model
        The model to save.
    path : str or Path
        Destination file path (e.g. ``models/retrained.keras``).

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    model.save(str(path))
    logger.info("Saved model: %s", path.name)
    return path


# ─── Predictions I/O ─────────────────────────────────────────────────

def load_predictions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load a predictions CSV in the standardised pipeline format.

    Expected columns include at least ``predicted_label`` and/or
    ``confidence``.

    Parameters
    ----------
    csv_path : str or Path
        Path to the predictions CSV.

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(csv_path)
    validate_file_exists(csv_path, "Predictions CSV")
    df = pd.read_csv(csv_path)
    logger.info("Loaded predictions: %s  rows=%d  cols=%s",
                csv_path.name, len(df), list(df.columns))
    return df


# ─── Pipeline config ─────────────────────────────────────────────────

def load_config_from_yaml(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load pipeline configuration from a YAML file.

    Parameters
    ----------
    path : str or Path, optional
        Defaults to ``config/pipeline_config.yaml`` relative to project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if path is None:
        # Resolve relative to this file → project root
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "config" / "pipeline_config.yaml"
    return read_yaml(path)


# ─── Class distribution ──────────────────────────────────────────────

def compute_class_distribution(
    predictions: np.ndarray,
    n_classes: int = 11,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute class frequency distribution from integer predictions.

    Parameters
    ----------
    predictions : np.ndarray
        1-D array of predicted class indices.
    n_classes : int
        Total number of classes (default 11).
    labels : list of str, optional
        Human-readable class names (default: activity labels).

    Returns
    -------
    dict
        ``{ label: count, ... , "_total": N }``.
    """
    if labels is None:
        labels = get_activity_labels()
    counts: Dict[str, Any] = {}
    for idx in range(n_classes):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        counts[label] = int(np.sum(predictions == idx))
    counts["_total"] = int(len(predictions))
    return counts


# ─── Logging helper ──────────────────────────────────────────────────

def setup_stage_logger(
    stage_name: str,
    log_dir: Union[str, Path],
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger that writes to both console and a stage-specific file.

    Parameters
    ----------
    stage_name : str
        Pipeline stage name (e.g. ``'ingestion'``).
    log_dir : str or Path
        Directory for log files.
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
    """
    log_dir = ensure_dir(log_dir)
    stage_logger = logging.getLogger(f"pipeline.{stage_name}")
    stage_logger.setLevel(level)

    # File handler (per-stage)
    fh = logging.FileHandler(
        log_dir / f"{stage_name}_{get_timestamp()}.log", encoding="utf-8"
    )
    fh.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    fh.setFormatter(fmt)
    stage_logger.addHandler(fh)
    return stage_logger


# ─── File archiving ──────────────────────────────────────────────────

def archive_file(
    src: Union[str, Path],
    archive_dir: Union[str, Path],
    suffix: Optional[str] = None,
) -> Path:
    """Copy *src* into *archive_dir* with a timestamp suffix.

    Parameters
    ----------
    src : str or Path
        Source file to archive.
    archive_dir : str or Path
        Target archive directory.
    suffix : str, optional
        Custom suffix (default: current timestamp).

    Returns
    -------
    Path
        Path of the archived copy.
    """
    src = Path(src)
    archive_dir = ensure_dir(archive_dir)
    if suffix is None:
        suffix = get_timestamp()
    dest = archive_dir / f"{src.stem}_{suffix}{src.suffix}"
    shutil.copy2(src, dest)
    logger.info("Archived %s → %s", src.name, dest)
    return dest
