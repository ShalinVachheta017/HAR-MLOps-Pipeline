"""
Component 8 – Model Retraining

Wraps:  src/train.py  →  HARTrainer / DomainAdaptationTrainer
        src/domain_adaptation/adabn.py  →  adapt_bn_statistics
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.entity.config_entity import ModelRetrainingConfig, PipelineConfig
from src.entity.artifact_entity import (
    TriggerEvaluationArtifact,
    DataTransformationArtifact,
    ModelRetrainingArtifact,
)

logger = logging.getLogger(__name__)


class ModelRetraining:
    """Retrain the model — standard, AdaBN, or pseudo-label adaptation."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: ModelRetrainingConfig,
        trigger_artifact: Optional[TriggerEvaluationArtifact] = None,
        transformation_artifact: Optional[DataTransformationArtifact] = None,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.trigger_artifact = trigger_artifact
        self.transformation_artifact = transformation_artifact

    # ------------------------------------------------------------------ #
    def initiate_model_retraining(self) -> ModelRetrainingArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 8 — Model Retraining")
        logger.info("=" * 60)

        output_dir = Path(
            self.config.output_dir
            or self.pipeline_config.models_dir / "retrained"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        method = self.config.adaptation_method if self.config.enable_adaptation else "none"
        logger.info("Retraining method: %s", method)

        if method == "adabn":
            return self._run_adabn(output_dir)
        elif method == "pseudo_label":
            return self._run_pseudo_label(output_dir)
        else:
            return self._run_standard(output_dir)

    # ------------------------------------------------------------------ #
    def _run_adabn(self, output_dir: Path) -> ModelRetrainingArtifact:
        """Unsupervised domain adaptation via AdaBN (no labels needed)."""
        from src.domain_adaptation.adabn import adapt_bn_statistics, adabn_score_confidence

        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required for AdaBN retraining.")

        # Load base model
        model_path = self.config.source_data_path  # repurposed or from pretrained
        base_model_path = (
            self.pipeline_config.models_pretrained_dir
            / "fine_tuned_model_1dcnnbilstm.keras"
        )
        logger.info("Loading base model: %s", base_model_path)
        model = keras.models.load_model(base_model_path)

        # Load target data
        target_npy = self.config.target_data_npy
        if target_npy is None and self.transformation_artifact:
            target_npy = self.transformation_artifact.production_X_path
        if target_npy is None:
            raise ValueError("AdaBN requires target_data_npy (production data).")

        target_X = np.load(target_npy)
        logger.info("Target data: %s", target_X.shape)

        # Before-adaptation confidence
        before = adabn_score_confidence(model, target_X)
        logger.info("Before AdaBN — mean confidence: %.4f", before["mean_confidence"])

        # Adapt
        model = adapt_bn_statistics(
            model,
            target_X,
            n_batches=self.config.adabn_n_batches,
            batch_size=self.config.batch_size,
        )

        # After-adaptation confidence
        after = adabn_score_confidence(model, target_X)
        logger.info("After AdaBN — mean confidence: %.4f", after["mean_confidence"])

        # Save adapted model
        save_path = output_dir / "adabn_adapted_model.keras"
        model.save(save_path)
        logger.info("Adapted model saved: %s", save_path)

        metrics = {
            "before_mean_confidence": before["mean_confidence"],
            "after_mean_confidence": after["mean_confidence"],
            "confidence_improvement": after["mean_confidence"] - before["mean_confidence"],
            "before_low_conf_ratio": before["low_confidence_ratio"],
            "after_low_conf_ratio": after["low_confidence_ratio"],
        }

        return ModelRetrainingArtifact(
            retrained_model_path=save_path,
            training_report={"before": before, "after": after},
            adaptation_method="adabn",
            metrics=metrics,
            n_target_samples=int(target_X.shape[0]),
            retraining_timestamp=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------ #
    def _run_pseudo_label(self, output_dir: Path) -> ModelRetrainingArtifact:
        """Semi-supervised: pseudo-label + fine-tune."""
        from src.train import DomainAdaptationTrainer, TrainingConfig

        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required for pseudo-label retraining.")

        train_cfg = TrainingConfig(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            enable_domain_adaptation=True,
            adaptation_method="pseudo_label",
            output_dir=str(output_dir),
            experiment_name=self.config.experiment_name,
        )

        trainer = DomainAdaptationTrainer(train_cfg)

        # Source data
        source_path = self.config.source_data_path or (
            self.pipeline_config.data_raw_dir / "all_users_data_labeled.csv"
        )
        from src.train import DataLoader as _DataLoader
        data_loader = _DataLoader(train_cfg, logger)
        source_X, source_y, _ = data_loader.prepare_data(str(source_path))

        # Target data
        target_npy = self.config.target_data_npy
        if target_npy is None and self.transformation_artifact:
            target_npy = self.transformation_artifact.production_X_path
        target_X = np.load(target_npy) if target_npy else None

        # Base model
        base_model_path = (
            self.pipeline_config.models_pretrained_dir
            / "fine_tuned_model_1dcnnbilstm.keras"
        )
        base_model = keras.models.load_model(base_model_path)

        model, metrics = trainer.retrain_with_adaptation(
            source_X, source_y, target_X, base_model,
        )

        save_path = output_dir / "pseudo_label_model.keras"
        model.save(save_path)

        return ModelRetrainingArtifact(
            retrained_model_path=save_path,
            training_report=metrics,
            adaptation_method="pseudo_label",
            metrics=metrics.get("final_metrics", {}),
            n_source_samples=int(source_X.shape[0]),
            n_target_samples=int(target_X.shape[0]) if target_X is not None else 0,
            retraining_timestamp=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------ #
    def _run_standard(self, output_dir: Path) -> ModelRetrainingArtifact:
        """Standard supervised retraining on labeled data."""
        from src.train import HARTrainer, TrainingConfig

        train_cfg = TrainingConfig(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            n_folds=self.config.n_folds,
            output_dir=str(output_dir),
            experiment_name=self.config.experiment_name,
        )

        trainer = HARTrainer(train_cfg)

        source_path = self.config.source_data_path or (
            self.pipeline_config.data_raw_dir / "all_users_data_labeled.csv"
        )

        from src.train import DataLoader as _DataLoader
        data_loader = _DataLoader(train_cfg, logger)
        X, y, _ = data_loader.prepare_data(str(source_path))

        # Cross-validation (optional)
        if not self.config.skip_cv:
            cv_results = trainer.run_cross_validation(X, y)
            logger.info("CV results: %s", cv_results)

        # Final model
        model, metrics = trainer.train_final_model(X, y, save_artifacts=True)

        save_path = output_dir / "retrained_model.keras"
        model.save(save_path)

        return ModelRetrainingArtifact(
            retrained_model_path=save_path,
            training_report=metrics,
            adaptation_method="none",
            metrics=metrics,
            n_source_samples=int(X.shape[0]),
            retraining_timestamp=datetime.now().isoformat(),
        )
