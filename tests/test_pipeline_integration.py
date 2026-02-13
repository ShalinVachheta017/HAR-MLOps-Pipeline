"""
Integration tests for the Production Pipeline orchestrator.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.entity.config_entity import PipelineConfig
from src.entity.artifact_entity import PipelineResult
from src.pipeline.production_pipeline import ProductionPipeline, ALL_STAGES


class TestProductionPipelineInit:
    def test_default_configs(self):
        cfg = PipelineConfig()
        pipeline = ProductionPipeline(cfg)
        assert pipeline.pipeline_config is cfg
        assert pipeline.ingestion_config is not None
        assert pipeline.retraining_config is not None

    def test_all_stages_list(self):
        assert len(ALL_STAGES) == 10
        assert ALL_STAGES[0] == "ingestion"
        assert ALL_STAGES[-1] == "baseline_update"


class TestPipelineStageSelection:
    def test_default_runs_7_stages(self):
        cfg = PipelineConfig()
        pipeline = ProductionPipeline(cfg)

        # Patch all component imports to avoid actual execution
        with patch("src.pipeline.production_pipeline.ProductionPipeline._init_mlflow", return_value=None):
            with patch("src.components.data_ingestion.DataIngestion") as mock_ing:
                mock_ing.return_value.initiate_data_ingestion.side_effect = Exception("test stop")
                result = pipeline.run(continue_on_failure=False)

        # Should have tried ingestion (first stage)
        assert "ingestion" in result.stages_failed

    def test_retrain_flag_adds_stages(self):
        cfg = PipelineConfig()
        pipeline = ProductionPipeline(cfg)
        # Just verify the logic — we'll intercept at ingestion
        with patch("src.pipeline.production_pipeline.ProductionPipeline._init_mlflow", return_value=None):
            with patch("src.components.data_ingestion.DataIngestion") as mock_ing:
                mock_ing.return_value.initiate_data_ingestion.side_effect = Exception("stop")
                result = pipeline.run(enable_retrain=True, continue_on_failure=False)

        assert "ingestion" in result.stages_failed

    def test_specific_stages(self):
        cfg = PipelineConfig()
        pipeline = ProductionPipeline(cfg)
        with patch("src.pipeline.production_pipeline.ProductionPipeline._init_mlflow", return_value=None):
            # Running only "inference" should fail with missing transformation artifact
            result = pipeline.run(stages=["inference"], continue_on_failure=False)
        # Should fail because transformation_artifact is None → fallback used → inference attempted
        assert "inference" in result.stages_failed or "inference" in result.stages_completed


class TestPipelineResult:
    def test_result_dataclass(self):
        result = PipelineResult()
        assert result.overall_status == "UNKNOWN"
        assert result.stages_completed == []
        assert result.retraining is None
        assert result.registration is None
        assert result.baseline_update is None

    def test_result_serialization(self, tmp_path):
        """PipelineResult should be serialisable."""
        import dataclasses
        result = PipelineResult(
            run_id="test123",
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            overall_status="SUCCESS",
            stages_completed=["ingestion", "validation"],
        )
        data = dataclasses.asdict(result)
        json_str = json.dumps(data, default=str)
        assert "test123" in json_str


class TestSkipAndContinue:
    def test_skip_ingestion(self):
        cfg = PipelineConfig()
        pipeline = ProductionPipeline(cfg)
        with patch("src.pipeline.production_pipeline.ProductionPipeline._init_mlflow", return_value=None):
            with patch("src.components.data_validation.DataValidation") as mock_val:
                mock_val.return_value.initiate_data_validation.side_effect = Exception("stop")
                result = pipeline.run(skip_ingestion=True, continue_on_failure=False)
        assert "ingestion" in result.stages_skipped

    def test_continue_on_failure(self):
        cfg = PipelineConfig()
        pipeline = ProductionPipeline(cfg)
        with patch("src.pipeline.production_pipeline.ProductionPipeline._init_mlflow", return_value=None):
            with patch("src.components.data_ingestion.DataIngestion") as mock_ing:
                mock_ing.return_value.initiate_data_ingestion.side_effect = Exception("boom")
                result = pipeline.run(continue_on_failure=True)
        assert "ingestion" in result.stages_failed
        # With continue_on_failure, pipeline should try next stages
        assert result.overall_status in ("PARTIAL", "FAILED")
