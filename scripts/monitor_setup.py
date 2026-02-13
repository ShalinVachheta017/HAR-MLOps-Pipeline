#!/usr/bin/env python3
"""
scripts/monitor_setup.py — Initialize monitoring infrastructure
================================================================

Sets up Prometheus, MLflow, and baseline statistics required by the
post-inference monitoring stage (stage 6).

Usage:
    python scripts/monitor_setup.py                    # interactive setup
    python scripts/monitor_setup.py --create-baselines # build drift baselines
    python scripts/monitor_setup.py --check            # verify monitoring is ready
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CONFIG_DIR, DATA_PREPARED, LOGS_DIR, MODELS_PRETRAINED,
    SENSOR_COLUMNS, ACTIVITY_LABELS, NUM_CLASSES,
)
from src.utils.common import ensure_dir, read_yaml, write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("monitor_setup")


def check_monitoring_readiness() -> bool:
    """Verify that all monitoring prerequisites are in place."""
    checks = []

    # 1. Config files
    prom = CONFIG_DIR / "prometheus.yml"
    mlflow_cfg = CONFIG_DIR / "mlflow_config.yaml"
    checks.append(("Prometheus config", prom.exists(), str(prom)))
    checks.append(("MLflow config", mlflow_cfg.exists(), str(mlflow_cfg)))

    # 2. Baseline stats
    baseline = DATA_PREPARED / "baseline_stats.json"
    checks.append(("Baseline stats", baseline.exists(), str(baseline)))

    # 3. Model file
    model = MODELS_PRETRAINED / "fine_tuned_model_1dcnnbilstm.keras"
    checks.append(("Pretrained model", model.exists(), str(model)))

    # 4. Log directories
    for sub in ("pipeline", "inference", "training", "evaluation", "preprocessing"):
        d = LOGS_DIR / sub
        checks.append((f"Log dir: {sub}", d.exists(), str(d)))

    # Report
    all_ok = True
    for name, ok, path in checks:
        status = "OK" if ok else "MISSING"
        marker = "+" if ok else "!"
        logger.info("  [%s] %-25s  %s  (%s)", marker, name, status, path)
        if not ok:
            all_ok = False

    return all_ok


def create_baselines():
    """Build baseline statistics from prepared data for drift detection."""
    import numpy as np

    production_x_path = DATA_PREPARED / "production_X.npy"
    if not production_x_path.exists():
        logger.error("Cannot create baselines — %s not found", production_x_path)
        logger.info("Run the preprocessing pipeline first:")
        logger.info("  python scripts/preprocess.py")
        return False

    logger.info("Loading production data: %s", production_x_path)
    X = np.load(production_x_path)  # (N, window_size, channels)
    logger.info("Shape: %s", X.shape)

    # Compute per-channel stats
    stats = {
        "n_windows": int(X.shape[0]),
        "window_size": int(X.shape[1]),
        "n_channels": int(X.shape[2]),
        "channels": {},
    }

    channel_names = SENSOR_COLUMNS if X.shape[2] == len(SENSOR_COLUMNS) else [
        f"ch_{i}" for i in range(X.shape[2])
    ]

    for i, name in enumerate(channel_names):
        ch_data = X[:, :, i].flatten()
        stats["channels"][name] = {
            "mean": float(np.mean(ch_data)),
            "std": float(np.std(ch_data)),
            "min": float(np.min(ch_data)),
            "max": float(np.max(ch_data)),
            "median": float(np.median(ch_data)),
            "q25": float(np.percentile(ch_data, 25)),
            "q75": float(np.percentile(ch_data, 75)),
        }

    out_path = DATA_PREPARED / "baseline_stats.json"
    write_json(out_path, stats)
    logger.info("Baseline stats saved: %s", out_path)
    return True


def create_log_dirs():
    """Ensure all log sub-directories exist."""
    for sub in ("pipeline", "inference", "training", "evaluation", "preprocessing"):
        ensure_dir(LOGS_DIR / sub)
    logger.info("Log directories ensured under %s", LOGS_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Set up monitoring infrastructure")
    parser.add_argument(
        "--check", action="store_true",
        help="Check monitoring readiness without modifying anything",
    )
    parser.add_argument(
        "--create-baselines", action="store_true",
        help="Build baseline statistics from prepared data",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("MONITORING SETUP")
    logger.info("=" * 60)

    # Always create log directories
    create_log_dirs()

    if args.create_baselines:
        create_baselines()

    if args.check or not args.create_baselines:
        logger.info("\nMonitoring readiness check:")
        ok = check_monitoring_readiness()
        if ok:
            logger.info("\nAll monitoring prerequisites are in place.")
        else:
            logger.warning("\nSome items are missing — see above.")

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
