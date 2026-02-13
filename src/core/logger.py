"""
Structured logging for HAR MLOps Pipeline.
Creates a single logger with both console and file output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def get_pipeline_logger(
    name: str = "har_pipeline",
    log_dir: Path = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create / retrieve a pipeline logger with console + file handlers.

    Args:
        name: Logger name (used as prefix in log files).
        log_dir: Directory for log files. Defaults to ``logs/pipeline/``.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = RotatingFileHandler(
        log_dir / f"{name}_{timestamp}.log",
        maxBytes=5_000_000,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
