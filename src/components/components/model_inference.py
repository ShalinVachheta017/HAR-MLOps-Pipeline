"""
Component 4 – Model Inference

Wraps:  src/run_inference.py  →  InferencePipeline (renamed to avoid clash)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.entity.config_entity import ModelInferenceConfig, PipelineConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelInferenceArtifact,
)

logger = logging.getLogger(__name__)


class ModelInference:
    """Run batch inference with the pretrained model."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: ModelInferenceConfig,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.transformation_artifact = transformation_artifact

    # ------------------------------------------------------------------ #
    def initiate_model_inference(self) -> ModelInferenceArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 4 — Model Inference")
        logger.info("=" * 60)

        from src.run_inference import (
            InferencePipeline as _InferencePipeline,
            InferenceConfig as _InferenceConfig,
        )

        # Build the internal config
        model_path = self.config.model_path or (
            self.pipeline_config.models_pretrained_dir
            / "fine_tuned_model_1dcnnbilstm.keras"
        )
        input_npy = self.config.input_npy or self.transformation_artifact.production_X_path
        output_dir = Path(
            self.config.output_dir
            or self.pipeline_config.data_prepared_dir / "predictions"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        inf_cfg = _InferenceConfig(
            model_path=Path(model_path),
            input_path=Path(input_npy),
            output_dir=output_dir,
            batch_size=self.config.batch_size,
            confidence_threshold=self.config.confidence_threshold,
            mode=self.config.mode,
        )

        pipe = _InferencePipeline(config=inf_cfg)
        result = pipe.run()

        # Parse result dict to artifact
        predictions_csv = Path(result.get(
            "predictions_csv",
            output_dir / "predictions_fresh.csv",
        ))
        predictions_npy = Path(result.get(
            "predictions_npy",
            output_dir / "production_predictions_fresh.npy",
        ))
        probabilities_npy = result.get("probabilities_npy")
        if probabilities_npy:
            probabilities_npy = Path(probabilities_npy)

        return ModelInferenceArtifact(
            predictions_csv_path=predictions_csv,
            predictions_npy_path=predictions_npy,
            probabilities_npy_path=probabilities_npy,
            n_predictions=result.get("n_predictions", 0),
            inference_time_seconds=result.get("inference_time", 0.0),
            model_version=result.get("model_version", ""),
        )
