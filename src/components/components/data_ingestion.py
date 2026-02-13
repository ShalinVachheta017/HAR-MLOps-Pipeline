"""
Component 1 – Data Ingestion

Wraps:  src/sensor_data_pipeline.py  (Excel Excel → fused CSV)
Also:   supports direct CSV input (user's own recordings, ABCD cases)
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.entity.config_entity import DataIngestionConfig, PipelineConfig
from src.entity.artifact_entity import DataIngestionArtifact

logger = logging.getLogger(__name__)


class DataIngestion:
    """Ingest raw sensor files (Excel pair or single CSV) into a fused CSV."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: DataIngestionConfig,
    ):
        self.pipeline_config = pipeline_config
        self.config = config

    # ------------------------------------------------------------------ #
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Run data ingestion and return the artifact."""
        logger.info("=" * 60)
        logger.info("STAGE 1 — Data Ingestion")
        logger.info("=" * 60)

        output_dir = self.config.output_dir or (
            self.pipeline_config.data_processed_dir
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Path A: direct CSV input ──────────────────────────────────
        if self.config.input_csv is not None:
            return self._ingest_csv(Path(self.config.input_csv), output_dir)

        # ── Path B: Excel pair (Garmin export) ────────────────────────
        return self._ingest_excel(output_dir)

    # ------------------------------------------------------------------ #
    def _ingest_csv(self, csv_path: Path, output_dir: Path) -> DataIngestionArtifact:
        """Copy / validate a user-provided CSV."""
        logger.info("Ingesting CSV: %s", csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        dest = output_dir / "sensor_fused_50Hz.csv"
        if csv_path.resolve() != dest.resolve():
            shutil.copy2(csv_path, dest)
        logger.info("CSV ingested: %d rows × %d cols → %s", len(df), len(df.columns), dest)

        return DataIngestionArtifact(
            fused_csv_path=dest,
            n_rows=len(df),
            n_columns=len(df.columns),
            sampling_hz=self.config.target_hz,
            ingestion_timestamp=datetime.now().isoformat(),
            source_type="csv",
        )

    # ------------------------------------------------------------------ #
    def _ingest_excel(self, output_dir: Path) -> DataIngestionArtifact:
        """Process a Garmin accel/gyro Excel pair via SensorDataPipeline."""
        from src.sensor_data_pipeline import (
            ProcessingConfig,
            LoggerSetup,
            SensorDataLoader,
            DataProcessor,
            SensorFusion,
            Resampler,
            find_latest_sensor_pair,
        )

        raw_dir = self.pipeline_config.data_raw_dir

        # Locate files
        if self.config.accel_file and self.config.gyro_file:
            accel_path = Path(self.config.accel_file)
            gyro_path = Path(self.config.gyro_file)
        else:
            accel_path, gyro_path = find_latest_sensor_pair(raw_dir)
        logger.info("Accel: %s  |  Gyro: %s", accel_path, gyro_path)

        # Run the pipeline components
        proc_cfg = ProcessingConfig(
            target_hz=self.config.target_hz,
            merge_tolerance_ms=self.config.merge_tolerance_ms,
        )
        log_setup = LoggerSetup(self.pipeline_config.logs_dir / "preprocessing")
        sdp_logger = log_setup.get_logger()

        loader = SensorDataLoader(sdp_logger)
        processor = DataProcessor(sdp_logger)
        fusion = SensorFusion(proc_cfg, sdp_logger)
        resampler = Resampler(proc_cfg, sdp_logger)

        # Load
        accel_df = loader.load_sensor_data(accel_path)
        gyro_df = loader.load_sensor_data(gyro_path)
        accel_df = loader.normalize_column_names(accel_df, "accelerometer")
        gyro_df = loader.normalize_column_names(gyro_df, "gyroscope")
        accel_df = loader.parse_list_columns(accel_df)
        gyro_df = loader.parse_list_columns(gyro_df)
        accel_df = loader.filter_valid_rows(accel_df)
        gyro_df = loader.filter_valid_rows(gyro_df)

        # Process
        accel_df = processor.process_sensor_data(accel_df, "accelerometer")
        gyro_df = processor.process_sensor_data(gyro_df, "gyroscope")

        # Merge + resample
        fused = fusion.merge_sensor_data(accel_df, gyro_df)
        fused = resampler.resample_data(fused)
        fused = resampler.add_timestamp_columns(fused)

        # Save
        dest = output_dir / "sensor_fused_50Hz.csv"
        fused.to_csv(dest, index=False)
        logger.info("Excel ingested: %d rows × %d cols → %s", len(fused), len(fused.columns), dest)

        return DataIngestionArtifact(
            fused_csv_path=dest,
            n_rows=len(fused),
            n_columns=len(fused.columns),
            sampling_hz=self.config.target_hz,
            ingestion_timestamp=datetime.now().isoformat(),
            source_type="excel",
        )
