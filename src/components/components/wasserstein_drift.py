"""
Component 12 – Wasserstein Drift Detection

Wraps:  src/wasserstein_drift.py  →  WassersteinDriftDetector,
        WassersteinChangePointDetector, compute_integrated_drift_report
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.entity.config_entity import WassersteinDriftConfig as WDConfig, PipelineConfig
from src.entity.artifact_entity import (
    PostInferenceMonitoringArtifact,
    DataTransformationArtifact,
    WassersteinDriftArtifact,
)

logger = logging.getLogger(__name__)


class WassersteinDrift:
    """Wasserstein distance-based distribution drift detection."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: WDConfig,
        transformation_artifact: DataTransformationArtifact,
        monitoring_artifact: Optional[PostInferenceMonitoringArtifact] = None,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.transformation_artifact = transformation_artifact
        self.monitoring_artifact = monitoring_artifact

    # ------------------------------------------------------------------ #
    def initiate_wasserstein_drift(self) -> WassersteinDriftArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 12 — Wasserstein Drift Detection")
        logger.info("=" * 60)

        from src.wasserstein_drift import (
            WassersteinDriftConfig as _WDCfg,
            WassersteinDriftDetector,
            compute_integrated_drift_report,
        )

        output_dir = Path(
            self.config.output_dir
            or self.pipeline_config.outputs_dir / "wasserstein_drift"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load production data
        prod_X_path = self.transformation_artifact.production_X_path
        if prod_X_path and Path(prod_X_path).exists():
            production_data = np.load(prod_X_path)
        else:
            logger.error("No production data found at %s", prod_X_path)
            return WassersteinDriftArtifact(overall_status="ERROR")

        # Load baseline data
        baseline_path = self.config.baseline_data_path
        if baseline_path is None:
            baseline_path = (
                self.pipeline_config.data_prepared_dir / "baseline_X.npy"
            )
        if not Path(baseline_path).exists():
            logger.warning(
                "No baseline data at %s — using production data as self-baseline.",
                baseline_path,
            )
            return WassersteinDriftArtifact(
                overall_status="NO_BASELINE",
                calibration_warnings=["No baseline data available for drift detection."],
            )

        baseline_data = np.load(baseline_path)

        # Initialize detector
        det_config = _WDCfg(
            warn_threshold=self.config.warn_threshold,
            critical_threshold=self.config.critical_threshold,
            min_drifted_channels_warn=self.config.min_drifted_channels_warn,
            min_drifted_channels_critical=self.config.min_drifted_channels_critical,
        )
        detector = WassersteinDriftDetector(det_config)

        # Run detection
        report = detector.detect(
            baseline_data,
            production_data,
            channel_names=self.config.sensor_columns,
        )

        # Run integrated report (PSI + KS + Wasserstein) if scipy available
        integrated = {}
        try:
            integrated = compute_integrated_drift_report(
                baseline_data,
                production_data,
                channel_names=self.config.sensor_columns,
            )
        except ImportError:
            logger.warning("scipy not available — skipping integrated report.")

        # Save report
        import json
        report_path = output_dir / "wasserstein_drift_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {"wasserstein": report, "integrated": integrated},
                f,
                indent=2,
                default=str,
            )

        return WassersteinDriftArtifact(
            overall_status=report["overall_status"],
            mean_wasserstein=report["mean_wasserstein"],
            max_wasserstein=report["max_wasserstein"],
            n_channels_warn=report["n_channels_warn"],
            n_channels_critical=report["n_channels_critical"],
            per_channel=report["per_channel"],
            drift_trend="UNKNOWN",
            integrated_report=integrated,
            report_path=report_path,
        )
