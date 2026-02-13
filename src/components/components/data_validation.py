"""
Component 2 – Data Validation

Wraps:  src/data_validator.py  →  DataValidator.validate()
"""

import logging
from pathlib import Path

import pandas as pd

from src.entity.config_entity import DataValidationConfig, PipelineConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

logger = logging.getLogger(__name__)


class DataValidation:
    """Validate the ingested sensor CSV (schema, ranges, completeness)."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: DataValidationConfig,
        ingestion_artifact: DataIngestionArtifact,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    # ------------------------------------------------------------------ #
    def initiate_data_validation(self) -> DataValidationArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 2 — Data Validation")
        logger.info("=" * 60)

        from src.data_validator import DataValidator

        csv_path = Path(self.ingestion_artifact.fused_csv_path)
        logger.info("Validating: %s", csv_path)
        df = pd.read_csv(csv_path)

        validator = DataValidator(
            sensor_columns=self.config.sensor_columns,
            expected_frequency_hz=self.config.expected_frequency_hz,
            max_acceleration=self.config.max_acceleration_ms2,
            max_gyroscope=self.config.max_gyroscope_dps,
            max_missing_ratio=self.config.max_missing_ratio,
        )
        result = validator.validate(df)

        logger.info("Validation result: valid=%s  errors=%d  warnings=%d",
                     result.is_valid, len(result.errors), len(result.warnings))

        return DataValidationArtifact(
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
            stats=result.stats,
        )
