"""
Component 10 – Baseline Update

Wraps:  scripts/build_training_baseline.py  →  BaselineBuilder
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.entity.config_entity import BaselineUpdateConfig, PipelineConfig
from src.entity.artifact_entity import (
    ModelRetrainingArtifact,
    BaselineUpdateArtifact,
)

logger = logging.getLogger(__name__)


class BaselineUpdate:
    """Rebuild drift baselines after retraining."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: BaselineUpdateConfig,
        retraining_artifact: Optional[ModelRetrainingArtifact] = None,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.retraining_artifact = retraining_artifact

    # ------------------------------------------------------------------ #
    def initiate_baseline_update(self) -> BaselineUpdateArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 10 — Baseline Update")
        logger.info("=" * 60)

        import sys
        scripts_dir = str(self.pipeline_config.scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from build_training_baseline import BaselineBuilder

        builder = BaselineBuilder()

        # Determine data source for baseline
        data_path = self.config.training_data_path or (
            self.pipeline_config.data_raw_dir / "all_users_data_labeled.csv"
        )
        logger.info("Building baseline from: %s", data_path)

        # Build
        baseline = builder.build_from_csv(data_path)

        # Save paths
        output_baseline = self.config.output_baseline_path or (
            self.pipeline_config.models_dir / "training_baseline.json"
        )
        output_normalized = self.config.output_normalized_path or (
            self.pipeline_config.models_dir / "normalized_baseline.json"
        )

        builder.save(output_baseline)
        logger.info("Baseline saved: %s", output_baseline)

        # Stats summary
        stats = {
            "n_channels": baseline.get("n_channels", 6),
            "n_samples": baseline.get("n_samples", 0),
            "activities": list(baseline.get("per_class", {}).keys()) if "per_class" in baseline else [],
        }

        return BaselineUpdateArtifact(
            baseline_path=Path(output_baseline),
            normalized_baseline_path=Path(output_normalized),
            n_channels=stats["n_channels"],
            stats_summary=stats,
            update_timestamp=datetime.now().isoformat(),
        )
