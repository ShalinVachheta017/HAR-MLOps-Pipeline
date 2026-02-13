"""
Component 14 – Sensor Placement Robustness

Wraps:  src/sensor_placement.py  →  AxisMirrorAugmenter, HandDetector,
        HandPerformanceReporter
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.entity.config_entity import SensorPlacementConfig as SPConfig, PipelineConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    SensorPlacementArtifact,
)

logger = logging.getLogger(__name__)


class SensorPlacement:
    """Sensor placement detection, axis mirroring augmentation, and per-hand reporting."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: SPConfig,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.transformation_artifact = transformation_artifact

    # ------------------------------------------------------------------ #
    def initiate_sensor_placement(self) -> SensorPlacementArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 14 — Sensor Placement Robustness")
        logger.info("=" * 60)

        from src.sensor_placement import (
            SensorPlacementConfig as _SPCfg,
            AxisMirrorAugmenter,
            HandDetector,
        )

        output_dir = Path(
            self.config.output_dir
            or self.pipeline_config.outputs_dir / "sensor_placement"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load production data
        prod_X_path = self.transformation_artifact.production_X_path
        if not Path(prod_X_path).exists():
            logger.error("Production data not found: %s", prod_X_path)
            return SensorPlacementArtifact()

        production_X = np.load(prod_X_path)
        n_original = len(production_X)

        # Configure
        sp_cfg = _SPCfg(
            mirror_axes=self.config.mirror_axes,
            mirror_probability=self.config.mirror_probability,
            dominant_accel_threshold=self.config.dominant_accel_threshold,
            accel_indices=self.config.accel_indices,
            gyro_indices=self.config.gyro_indices,
        )

        # --- Hand Detection ---
        detector = HandDetector(sp_cfg)
        hand_info = detector.detect(production_X)

        logger.info(
            "Hand detection: %s (confidence=%.2f)",
            hand_info["detected_hand"],
            hand_info["detection_confidence"],
        )

        # --- Axis Mirroring Augmentation ---
        # Only augment if used for training data enrichment
        augmenter = AxisMirrorAugmenter(sp_cfg)

        # Save hand detection results
        import json
        hand_report_path = output_dir / "hand_detection.json"
        with open(hand_report_path, "w") as f:
            json.dump(hand_info, f, indent=2, default=str)

        return SensorPlacementArtifact(
            detected_hand=hand_info["detected_hand"],
            detection_confidence=hand_info["detection_confidence"],
            n_original_samples=n_original,
            hand_features=hand_info.get("features", {}),
            per_hand_report=hand_info,
        )
