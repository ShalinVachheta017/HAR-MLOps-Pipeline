"""
Component 13 – Curriculum Pseudo-Labeling

Wraps:  src/curriculum_pseudo_labeling.py  →  CurriculumTrainer
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.entity.config_entity import CurriculumPseudoLabelingConfig, PipelineConfig
from src.entity.artifact_entity import (
    ModelRetrainingArtifact,
    DataTransformationArtifact,
    CurriculumPseudoLabelingArtifact,
)

logger = logging.getLogger(__name__)


class CurriculumPseudoLabeling:
    """Self-training with progressive pseudo-labeling and EWC."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: CurriculumPseudoLabelingConfig,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.transformation_artifact = transformation_artifact

    # ------------------------------------------------------------------ #
    def initiate_curriculum_training(self) -> CurriculumPseudoLabelingArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 13 — Curriculum Pseudo-Labeling")
        logger.info("=" * 60)

        from src.curriculum_pseudo_labeling import (
            CurriculumConfig as _CConfig,
            CurriculumTrainer,
        )

        output_dir = Path(
            self.config.output_dir
            or self.pipeline_config.outputs_dir / "curriculum_training"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load source (labeled) data
        source_path = self.config.source_data_path
        if source_path is None:
            source_path = self.pipeline_config.data_prepared_dir / "train_X.npy"
        labels_path = source_path.parent / "train_y.npy"

        if not Path(source_path).exists() or not Path(labels_path).exists():
            logger.error("Source labeled data not found: %s", source_path)
            return CurriculumPseudoLabelingArtifact()

        labeled_X = np.load(source_path)
        labeled_y = np.load(labels_path)

        # Load unlabeled production data
        unlabeled_path = (
            self.config.unlabeled_data_path
            or self.transformation_artifact.production_X_path
        )
        if not Path(unlabeled_path).exists():
            logger.error("Unlabeled data not found: %s", unlabeled_path)
            return CurriculumPseudoLabelingArtifact()

        unlabeled_X = np.load(unlabeled_path)

        # Load model
        try:
            import tensorflow as tf
            model_path = self.pipeline_config.models_pretrained_dir / "model.h5"
            if not model_path.exists():
                model_path = self.pipeline_config.models_pretrained_dir / "model.keras"
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return CurriculumPseudoLabelingArtifact()

        # Configure trainer
        train_cfg = _CConfig(
            initial_confidence_threshold=self.config.initial_confidence_threshold,
            final_confidence_threshold=self.config.final_confidence_threshold,
            n_iterations=self.config.n_iterations,
            threshold_decay=self.config.threshold_decay,
            max_samples_per_class=self.config.max_samples_per_class,
            min_samples_per_class=self.config.min_samples_per_class,
            use_teacher_student=self.config.use_teacher_student,
            ema_decay=self.config.ema_decay,
            use_ewc=self.config.use_ewc,
            ewc_lambda=self.config.ewc_lambda,
            ewc_n_samples=self.config.ewc_n_samples,
            epochs_per_iteration=self.config.epochs_per_iteration,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )

        trainer = CurriculumTrainer(train_cfg)
        result = trainer.train(
            model=model,
            labeled_X=labeled_X,
            labeled_y=labeled_y,
            unlabeled_X=unlabeled_X,
        )

        # Save retrained model
        retrained_path = output_dir / "curriculum_retrained_model.keras"
        result["model"].save(retrained_path)

        total_pseudo = sum(
            log.get("n_selected", 0)
            for log in result["iteration_logs"]
            if not log.get("skipped", False)
        )

        return CurriculumPseudoLabelingArtifact(
            retrained_model_path=retrained_path,
            iterations_completed=result["iterations_completed"],
            total_pseudo_labeled=total_pseudo,
            best_val_accuracy=result["best_val_accuracy"],
            final_mean_confidence=result["final_mean_confidence"],
            iteration_logs=result["iteration_logs"],
            ewc_used=self.config.use_ewc,
            training_report=result,
        )
