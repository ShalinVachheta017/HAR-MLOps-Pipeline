#!/usr/bin/env python3
"""
scripts/train.py — Standalone training / retraining entry point
================================================================

Runs model retraining (stage 8) with optional AdaBN domain adaptation,
model registration (stage 9), and baseline update (stage 10).

Usage:
    python scripts/train.py
    python scripts/train.py --adapt adabn --epochs 50
    python scripts/train.py --adapt pseudo_label --epochs 100
    python scripts/train.py --labels data/prepared/ground_truth.npy
    python scripts/train.py --auto-deploy
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.entity.config_entity import (
    PipelineConfig,
    ModelRetrainingConfig,
    ModelRegistrationConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run retraining → registration → baseline update (stages 8-10)",
    )
    parser.add_argument(
        "--adapt",
        type=str,
        choices=["adabn", "pseudo_label", "none"],
        default="none",
        help="Domain adaptation method (default: none)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to ground-truth labels for supervised retraining",
    )
    parser.add_argument(
        "--auto-deploy",
        action="store_true",
        help="Automatically deploy retrained model if validation passes",
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Skip model registration (stage 9)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline update (stage 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Stage 8: Model Retraining ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 8: Model Retraining  (adapt=%s, epochs=%d)", args.adapt, args.epochs)
    logger.info("=" * 60)

    from src.components.model_retraining import ModelRetraining

    retraining_cfg = ModelRetrainingConfig(
        enable_adaptation=(args.adapt != "none"),
        adaptation_method=args.adapt,
        epochs=args.epochs,
        labels_path=Path(args.labels) if args.labels else None,
    )
    retraining = ModelRetraining(config=retraining_cfg)
    retraining_artifact = retraining.initiate_model_retraining()
    logger.info("Retraining complete: %s", retraining_artifact)

    # ── Stage 9: Model Registration ───────────────────────────────────
    if not args.skip_registration:
        logger.info("=" * 60)
        logger.info("STAGE 9: Model Registration")
        logger.info("=" * 60)

        from src.components.model_registration import ModelRegistration

        registration_cfg = ModelRegistrationConfig(auto_deploy=args.auto_deploy)
        registration = ModelRegistration(
            config=registration_cfg, prev_artifact=retraining_artifact,
        )
        registration_artifact = registration.initiate_model_registration()
        logger.info("Registration complete: %s", registration_artifact)
    else:
        logger.info("Skipping model registration")

    # ── Stage 10: Baseline Update ─────────────────────────────────────
    if not args.skip_baseline:
        logger.info("=" * 60)
        logger.info("STAGE 10: Baseline Update")
        logger.info("=" * 60)

        from src.components.baseline_update import BaselineUpdate
        from src.entity.config_entity import BaselineUpdateConfig

        baseline_cfg = BaselineUpdateConfig()
        baseline = BaselineUpdate(config=baseline_cfg)
        baseline_artifact = baseline.initiate_baseline_update()
        logger.info("Baseline update complete: %s", baseline_artifact)
    else:
        logger.info("Skipping baseline update")

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
