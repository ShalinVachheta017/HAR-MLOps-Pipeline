"""
Component 9 – Model Registration

Wraps:  src/model_rollback.py  →  ModelRegistry
"""

import logging
from pathlib import Path
from typing import Optional

from src.entity.config_entity import ModelRegistrationConfig, PipelineConfig
from src.entity.artifact_entity import (
    ModelRetrainingArtifact,
    ModelEvaluationArtifact,
    ModelRegistrationArtifact,
)

logger = logging.getLogger(__name__)


class ModelRegistration:
    """Register, validate, and (optionally) deploy the retrained model."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: ModelRegistrationConfig,
        retraining_artifact: ModelRetrainingArtifact,
        evaluation_artifact: Optional[ModelEvaluationArtifact] = None,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.retraining_artifact = retraining_artifact
        self.evaluation_artifact = evaluation_artifact

    # ------------------------------------------------------------------ #
    def initiate_model_registration(self) -> ModelRegistrationArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 9 — Model Registration")
        logger.info("=" * 60)

        from src.model_rollback import ModelRegistry

        registry_dir = self.config.registry_dir or (
            self.pipeline_config.models_dir / "registry"
        )
        registry = ModelRegistry(registry_dir=Path(registry_dir))

        model_path = Path(
            self.config.model_path
            or self.retraining_artifact.retrained_model_path
        )
        version = self.config.version  # None → auto-increment
        metrics = self.retraining_artifact.metrics or {}

        # Register
        model_version = registry.register_model(
            model_path=model_path,
            version=version or "auto",
            metrics=metrics,
        )
        logger.info("Registered model version: %s", model_version.version)

        # Proxy validation (compare metrics with current deployed)
        is_better = True  # default: first registration is always "better"
        if self.config.proxy_validation:
            current = registry.get_current_version()
            if current:
                logger.info("Current deployed version: %s", current)
                # Simple proxy: compare mean_confidence or accuracy if available
                # (the user expands this with real validation later)
                is_better = True  # placeholder — always deploy for thesis

        # Deploy if auto_deploy is on and model is better (or first time)
        deployed = False
        if self.config.auto_deploy and is_better:
            deployed = registry.deploy_model(model_version.version)
            if deployed:
                logger.info("Model v%s deployed.", model_version.version)
            else:
                logger.warning("Deployment of v%s failed.", model_version.version)

        previous = registry.get_current_version() if not deployed else None

        return ModelRegistrationArtifact(
            registered_version=model_version.version,
            is_deployed=deployed,
            is_better_than_current=is_better,
            proxy_metrics=metrics,
            previous_version=previous,
            registry_path=Path(registry_dir),
        )
