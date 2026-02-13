"""
Tests for the Baseline Update component (stage 10).
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.entity.config_entity import PipelineConfig, BaselineUpdateConfig
from src.entity.artifact_entity import BaselineUpdateArtifact


@pytest.fixture
def pipeline_config(tmp_path):
    cfg = PipelineConfig()
    cfg.project_root = tmp_path
    cfg.models_dir = tmp_path / "models"
    cfg.data_raw_dir = tmp_path / "data" / "raw"
    cfg.scripts_dir = tmp_path / "scripts"
    cfg.logs_dir = tmp_path / "logs"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_raw_dir.mkdir(parents=True, exist_ok=True)
    return cfg


class TestBaselineUpdateConfig:
    def test_defaults(self):
        cfg = BaselineUpdateConfig()
        assert cfg.training_data_path is None
        assert cfg.rebuild_embeddings is False


class TestBaselineUpdateComponent:
    def test_initiate_with_mock_builder(self, pipeline_config, tmp_path):
        """Mock the BaselineBuilder to test the component wrapper logic."""
        # Create a fake training CSV
        import pandas as pd
        csv_path = pipeline_config.data_raw_dir / "all_users_data_labeled.csv"
        df = pd.DataFrame({
            "timestamp": range(100),
            "Ax_w": np.random.randn(100),
            "Ay_w": np.random.randn(100),
            "Az_w": np.random.randn(100),
            "Gx_w": np.random.randn(100),
            "Gy_w": np.random.randn(100),
            "Gz_w": np.random.randn(100),
            "activity": ["sitting"] * 50 + ["standing"] * 50,
            "User": [1] * 100,
        })
        df.to_csv(csv_path, index=False)

        # Mock BaselineBuilder
        mock_builder = MagicMock()
        mock_builder.build_from_csv.return_value = {
            "n_channels": 6,
            "n_samples": 100,
            "per_class": {"sitting": {}, "standing": {}},
        }

        config = BaselineUpdateConfig(
            output_baseline_path=tmp_path / "baseline.json",
            output_normalized_path=tmp_path / "normalized.json",
        )

        with patch.dict("sys.modules", {"build_training_baseline": MagicMock()}):
            import sys
            mock_module = MagicMock()
            mock_module.BaselineBuilder.return_value = mock_builder

            # Add scripts to path and patch
            scripts_dir = str(pipeline_config.scripts_dir)
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)

            with patch.dict("sys.modules", {"build_training_baseline": mock_module}):
                from src.components.baseline_update import BaselineUpdate
                comp = BaselineUpdate(pipeline_config, config)
                artifact = comp.initiate_baseline_update()

        assert isinstance(artifact, BaselineUpdateArtifact)
        assert artifact.n_channels == 6
        assert artifact.update_timestamp != ""
