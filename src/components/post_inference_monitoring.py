"""
Component 6 – Post-Inference Monitoring

Wraps:  scripts/post_inference_monitoring.py  →  PostInferenceMonitor
"""

import logging
from pathlib import Path
from typing import Optional

from src.entity.config_entity import PostInferenceMonitoringConfig, PipelineConfig
from src.entity.artifact_entity import (
    ModelInferenceArtifact,
    DataTransformationArtifact,
    PostInferenceMonitoringArtifact,
)

logger = logging.getLogger(__name__)


class PostInferenceMonitoring:
    """Run 3-layer monitoring (confidence, temporal, drift)."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: PostInferenceMonitoringConfig,
        inference_artifact: ModelInferenceArtifact,
        transformation_artifact: Optional[DataTransformationArtifact] = None,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.inference_artifact = inference_artifact
        self.transformation_artifact = transformation_artifact

    # ------------------------------------------------------------------ #
    def initiate_post_inference_monitoring(self) -> PostInferenceMonitoringArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 6 — Post-Inference Monitoring")
        logger.info("=" * 60)

        import sys, os
        # Ensure scripts/ is importable
        scripts_dir = str(self.pipeline_config.scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from post_inference_monitoring import PostInferenceMonitor

        monitor = PostInferenceMonitor()

        predictions_csv = Path(
            self.config.predictions_csv
            or self.inference_artifact.predictions_csv_path
        )
        production_npy = self.config.production_data_npy
        if production_npy is None and self.transformation_artifact:
            production_npy = self.transformation_artifact.production_X_path
        baseline_json = self.config.baseline_stats_json or (
            self.pipeline_config.models_dir / "normalized_baseline.json"
        )
        model_path = self.config.model_path or (
            self.pipeline_config.models_pretrained_dir
            / "fine_tuned_model_1dcnnbilstm.keras"
        )
        output_dir = Path(
            self.config.output_dir
            or self.pipeline_config.outputs_dir / "evaluation"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        report = monitor.run(
            predictions_path=predictions_csv,
            production_data_path=production_npy,
            baseline_path=baseline_json,
            model_path=model_path,
            output_dir=output_dir,
        )

        # Convert MonitoringReport → artifact
        report_dict = {}
        if hasattr(report, "__dict__"):
            report_dict = {
                k: v for k, v in report.__dict__.items()
                if not k.startswith("_")
            }

        return PostInferenceMonitoringArtifact(
            monitoring_report=report_dict,
            overall_status=getattr(report, "overall_status", "UNKNOWN"),
            layer1_confidence=getattr(report, "layer1_confidence", {}),
            layer2_temporal=getattr(report, "layer2_temporal", {}),
            layer3_drift=getattr(report, "layer3_drift", {}),
            report_path=output_dir / "monitoring_report.json",
        )
