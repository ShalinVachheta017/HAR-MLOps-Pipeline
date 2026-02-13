"""
Tests for the Model Retraining component (stage 8).
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.entity.config_entity import PipelineConfig, ModelRetrainingConfig
from src.entity.artifact_entity import (
    ModelRetrainingArtifact,
    DataTransformationArtifact,
    TriggerEvaluationArtifact,
)


@pytest.fixture
def pipeline_config(tmp_path):
    cfg = PipelineConfig()
    cfg.project_root = tmp_path
    cfg.models_dir = tmp_path / "models"
    cfg.models_pretrained_dir = tmp_path / "models" / "pretrained"
    cfg.data_raw_dir = tmp_path / "data" / "raw"
    cfg.data_prepared_dir = tmp_path / "data" / "prepared"
    cfg.logs_dir = tmp_path / "logs"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_pretrained_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_prepared_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def dummy_npy(tmp_path):
    """Create a dummy .npy target file."""
    path = tmp_path / "data" / "prepared" / "production_X.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.random.randn(50, 200, 6).astype(np.float32))
    return path


@pytest.fixture
def transformation_artifact(dummy_npy):
    return DataTransformationArtifact(
        production_X_path=dummy_npy,
        metadata_path=dummy_npy.parent / "meta.json",
        n_windows=50,
        window_size=200,
        unit_conversion_applied=False,
        preprocessing_timestamp=datetime.now().isoformat(),
    )


class TestModelRetrainingInit:
    def test_creates_instance(self, pipeline_config):
        from src.components.model_retraining import ModelRetraining
        config = ModelRetrainingConfig()
        comp = ModelRetraining(pipeline_config, config)
        assert comp.config.adaptation_method == "adabn"

    def test_adaptation_disabled_by_default(self, pipeline_config):
        config = ModelRetrainingConfig()
        assert config.enable_adaptation is False


class TestModelRetrainingAdaBN:
    @pytest.mark.skipif(
        not pytest.importorskip("tensorflow", reason="TF not installed"),
        reason="TF required",
    )
    def test_adabn_retraining(self, pipeline_config, transformation_artifact):
        """Integration-level test: AdaBN with a real tiny model."""
        tf = pytest.importorskip("tensorflow")
        keras = tf.keras

        # Create a tiny model
        pretrained_dir = pipeline_config.models_pretrained_dir
        inputs = keras.Input(shape=(200, 6))
        x = keras.layers.Conv1D(8, 3, padding="same")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(11, activation="softmax")(x)
        model = keras.Model(inputs, x)
        model.compile(optimizer="adam", loss="categorical_crossentropy")
        model.save(pretrained_dir / "fine_tuned_model_1dcnnbilstm.keras")

        from src.components.model_retraining import ModelRetraining
        config = ModelRetrainingConfig(
            enable_adaptation=True,
            adaptation_method="adabn",
            adabn_n_batches=2,
            batch_size=16,
        )
        comp = ModelRetraining(
            pipeline_config, config,
            transformation_artifact=transformation_artifact,
        )
        artifact = comp.initiate_model_retraining()

        assert isinstance(artifact, ModelRetrainingArtifact)
        assert artifact.adaptation_method == "adabn"
        assert artifact.retrained_model_path.exists()
        assert "after_mean_confidence" in artifact.metrics
