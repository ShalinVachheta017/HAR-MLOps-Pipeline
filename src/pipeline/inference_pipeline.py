"""
HAR MLOps Production Inference Pipeline
========================================

Single-file orchestrator that chains all pipeline components together.
Run the entire offline/analog pipeline with ONE command:

    python run_pipeline.py                      # full pipeline
    python run_pipeline.py --stages preprocess infer evaluate monitor
    python run_pipeline.py --skip-ingestion     # start from existing CSV

Pipeline Stages (in order):
    1. Data Ingestion      →  raw Excel → sensor_fused_50Hz.csv
    2. Data Validation     →  schema + range checks
    3. Preprocessing       →  CSV → windowed production_X.npy
    4. Inference            →  .npy + model → predictions CSV/NPY
    5. Evaluation          →  confidence / distribution analysis
    6. Monitoring          →  3-layer (confidence, temporal, drift)
    7. Trigger Evaluation  →  retraining decision

Architecture follows the reference-project pattern (see docs/thesis/production refrencxe/):
    entity/config_entity.py   →  @dataclass configs per stage
    entity/artifact_entity.py →  @dataclass artifacts passed between stages
    pipeline/inference_pipeline.py  →  this file (orchestrator)

Author: HAR MLOps Pipeline
Date: February 2026
"""

import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# ── Ensure src/ is importable ──────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Core utilities ─────────────────────────────────────────────────────────
from src.core.logger import get_pipeline_logger
from src.core.exception import PipelineException

# ── Entity dataclasses ─────────────────────────────────────────────────────
from src.entity.config_entity import (
    PipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    PreprocessingConfig,
    InferenceConfig,
    EvaluationConfig,
    MonitoringConfig,
    TriggerConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    PreprocessingArtifact,
    InferenceArtifact,
    EvaluationArtifact,
    MonitoringArtifact,
    TriggerArtifact,
    PipelineResult,
)

# ── Existing src modules (imported lazily inside each stage method) ────────
# This avoids import-time side effects (TensorFlow, MLflow, etc.)
# and lets the pipeline start fast even when only running early stages.


logger = get_pipeline_logger("inference_pipeline")


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """
    Production inference pipeline — chains all stages with artifact passing.

    Each ``start_*`` method:
        1. Receives the previous stage's artifact
        2. Instantiates the real component from ``src/``
        3. Returns a new artifact
    Failures are caught and wrapped in ``PipelineException`` with stage context.
    """

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None):
        self.cfg = pipeline_config or PipelineConfig()
        self.result = PipelineResult(
            run_id=self.cfg.timestamp,
            start_time=datetime.now().isoformat(),
        )
        logger.info("=" * 70)
        logger.info("HAR MLOps PRODUCTION PIPELINE")
        logger.info(f"  Run ID   : {self.cfg.timestamp}")
        logger.info(f"  Project  : {self.cfg.project_root}")
        logger.info("=" * 70)

    # ────────────────────────────────────────────────────────────────────
    # STAGE 1: Data Ingestion  (Excel → fused CSV)
    # ────────────────────────────────────────────────────────────────────
    def start_data_ingestion(
        self, config: Optional[DataIngestionConfig] = None
    ) -> DataIngestionArtifact:
        """Load raw Garmin Excel files, fuse sensors, resample → CSV."""
        stage = "data_ingestion"
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE 1: DATA INGESTION")
        logger.info(f"{'='*60}")
        t0 = time.time()

        try:
            from sensor_data_pipeline import SensorDataPipeline, find_latest_sensor_pair

            cfg = config or DataIngestionConfig()
            raw_dir = self.cfg.data_raw_dir

            # Auto-detect sensor files if not specified
            if cfg.accel_file and cfg.gyro_file:
                accel, gyro = cfg.accel_file, cfg.gyro_file
            else:
                accel, gyro = find_latest_sensor_pair(raw_dir)
                logger.info(f"  Auto-detected accel : {accel.name}")
                logger.info(f"  Auto-detected gyro  : {gyro.name}")

            # Run the sensor data pipeline
            pipeline = SensorDataPipeline(self.cfg.project_root)
            pipeline.process_sensor_files(accel, gyro)

            # Locate output
            fused_csv = self.cfg.data_processed_dir / "sensor_fused_50Hz.csv"
            if not fused_csv.exists():
                raise FileNotFoundError(f"Expected output not found: {fused_csv}")

            import pandas as pd
            df_info = pd.read_csv(fused_csv, nrows=5)

            artifact = DataIngestionArtifact(
                fused_csv_path=fused_csv,
                n_rows=sum(1 for _ in open(fused_csv)) - 1,  # exclude header
                n_columns=len(df_info.columns),
                sampling_hz=cfg.target_hz,
                ingestion_timestamp=datetime.now().isoformat(),
            )

            self.result.ingestion = artifact
            self.result.stages_completed.append(stage)
            logger.info(f"  Rows     : {artifact.n_rows:,}")
            logger.info(f"  Columns  : {artifact.n_columns}")
            logger.info(f"  Duration : {time.time()-t0:.1f}s")
            return artifact

        except Exception as e:
            self.result.stages_failed.append(stage)
            raise PipelineException(e, sys, stage=stage) from e

    # ────────────────────────────────────────────────────────────────────
    # STAGE 2: Data Validation
    # ────────────────────────────────────────────────────────────────────
    def start_data_validation(
        self,
        ingestion_artifact: DataIngestionArtifact,
        config: Optional[DataValidationConfig] = None,
    ) -> DataValidationArtifact:
        """Validate schema, ranges, missing values on the fused CSV."""
        stage = "data_validation"
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE 2: DATA VALIDATION")
        logger.info(f"{'='*60}")
        t0 = time.time()

        try:
            import pandas as pd
            from data_validator import DataValidator

            cfg = config or DataValidationConfig()
            df = pd.read_csv(ingestion_artifact.fused_csv_path)

            validator = DataValidator(
                sensor_columns=cfg.sensor_columns,
                expected_frequency_hz=cfg.expected_frequency_hz,
                max_acceleration=cfg.max_acceleration_ms2,
                max_gyroscope=cfg.max_gyroscope_dps,
                max_missing_ratio=cfg.max_missing_ratio,
            )
            result = validator.validate(df)

            artifact = DataValidationArtifact(
                is_valid=result.is_valid,
                errors=result.errors,
                warnings=result.warnings,
                stats=result.stats,
            )

            self.result.validation = artifact
            self.result.stages_completed.append(stage)

            status = "PASSED" if artifact.is_valid else "FAILED"
            logger.info(f"  Status   : {status}")
            if artifact.warnings:
                for w in artifact.warnings:
                    logger.warning(f"  Warning  : {w}")
            if artifact.errors:
                for e in artifact.errors:
                    logger.error(f"  Error    : {e}")
            logger.info(f"  Duration : {time.time()-t0:.1f}s")
            return artifact

        except Exception as e:
            self.result.stages_failed.append(stage)
            raise PipelineException(e, sys, stage=stage) from e

    # ────────────────────────────────────────────────────────────────────
    # STAGE 3: Preprocessing  (CSV → windowed .npy)
    # ────────────────────────────────────────────────────────────────────
    def start_preprocessing(
        self,
        ingestion_artifact: DataIngestionArtifact,
        config: Optional[PreprocessingConfig] = None,
    ) -> PreprocessingArtifact:
        """Unit detection, optional gravity removal / calibration, normalization, windowing."""
        stage = "preprocessing"
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE 3: PREPROCESSING")
        logger.info(f"{'='*60}")
        t0 = time.time()

        try:
            import pandas as pd
            import numpy as np
            from preprocess_data import (
                UnitDetector,
                GravityRemover,
                DomainCalibrator,
                UnifiedPreprocessor,
                PreprocessLogger,
            )

            cfg = config or PreprocessingConfig()
            input_csv = cfg.input_csv or ingestion_artifact.fused_csv_path

            plogger = PreprocessLogger("production_preprocessing")
            plog = plogger.get_logger()

            df = pd.read_csv(input_csv)
            logger.info(f"  Input    : {input_csv} ({len(df):,} rows)")

            # Initialize components
            unit_detector = UnitDetector(plog)
            gravity_remover = GravityRemover(plog)
            calibrator = DomainCalibrator(plog)
            preprocessor = UnifiedPreprocessor(plog)

            # Detect format
            data_format, sensor_cols = preprocessor.detect_data_format(df)
            accel_cols = [c for c in sensor_cols if c.startswith("A")]

            # Unit conversion
            df, conversion_applied = unit_detector.process_units(df, accel_cols)

            # Gravity removal or calibration (mutually exclusive)
            if cfg.enable_gravity_removal:
                df = gravity_remover.remove_gravity(df, enable=True)
            elif cfg.enable_calibration:
                df = calibrator.calibrate(df, enable=True)

            # Normalize
            df_norm = preprocessor.normalize_data(df, sensor_cols, mode="transform")

            # Create windows
            X, _, metadata = preprocessor.create_windows(df_norm, sensor_cols, data_format)

            # Save
            data = {"X": X}
            preprocessor.save_data(data, metadata, data_format, conversion_applied)

            # Locate saved files
            prepared_dir = self.cfg.data_prepared_dir
            production_X = prepared_dir / "production_X.npy"
            metadata_file = prepared_dir / "production_metadata.json"

            artifact = PreprocessingArtifact(
                production_X_path=production_X,
                metadata_path=metadata_file,
                n_windows=X.shape[0],
                window_size=X.shape[1],
                unit_conversion_applied=conversion_applied,
                preprocessing_timestamp=datetime.now().isoformat(),
            )

            self.result.preprocessing = artifact
            self.result.stages_completed.append(stage)
            logger.info(f"  Windows  : {artifact.n_windows:,}")
            logger.info(f"  Shape    : {X.shape}")
            logger.info(f"  Converted: {conversion_applied}")
            logger.info(f"  Duration : {time.time()-t0:.1f}s")
            return artifact

        except Exception as e:
            self.result.stages_failed.append(stage)
            raise PipelineException(e, sys, stage=stage) from e

    # ────────────────────────────────────────────────────────────────────
    # STAGE 4: Inference  (.npy + model → predictions)
    # ────────────────────────────────────────────────────────────────────
    def start_inference(
        self,
        preprocessing_artifact: PreprocessingArtifact,
        config: Optional[InferenceConfig] = None,
    ) -> InferenceArtifact:
        """Run batch inference with the pretrained 1D-CNN-BiLSTM model."""
        stage = "inference"
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE 4: INFERENCE")
        logger.info(f"{'='*60}")
        t0 = time.time()

        try:
            from run_inference import InferencePipeline as _InferencePipeline
            from run_inference import InferenceConfig as _InfCfg

            cfg = config or InferenceConfig()

            inf_config = _InfCfg(
                mode=cfg.mode,
                batch_size=cfg.batch_size,
                confidence_threshold=cfg.confidence_threshold,
            )
            # Override paths
            if cfg.input_npy:
                inf_config.input_path = cfg.input_npy
            else:
                inf_config.input_path = preprocessing_artifact.production_X_path

            if cfg.model_path:
                inf_config.model_path = cfg.model_path
            # else defaults to PRETRAINED_MODEL from config.py

            if cfg.output_dir:
                inf_config.output_dir = cfg.output_dir
            else:
                inf_config.output_dir = self.cfg.outputs_dir

            pipe = _InferencePipeline(inf_config)
            result = pipe.run()

            duration = time.time() - t0

            # Extract output file paths from the result
            n_preds = 0
            output_files = result.get("output_files", {})
            results_df = result.get("results")
            if results_df is not None and hasattr(results_df, "__len__"):
                n_preds = len(results_df)

            pred_csv = output_files.get("csv", cfg.output_dir or self.cfg.outputs_dir / "predictions_fresh.csv")
            pred_npy = output_files.get("npy", cfg.output_dir or self.cfg.outputs_dir / "production_predictions_fresh.npy")
            prob_npy = output_files.get("probabilities")

            artifact = InferenceArtifact(
                predictions_csv_path=Path(pred_csv),
                predictions_npy_path=Path(pred_npy),
                probabilities_npy_path=Path(prob_npy) if prob_npy else None,
                n_predictions=n_preds,
                inference_time_seconds=duration,
                model_version=result.get("mlflow_run_id", "pretrained"),
            )

            self.result.inference = artifact
            self.result.stages_completed.append(stage)
            logger.info(f"  Predictions : {artifact.n_predictions}")
            logger.info(f"  Output CSV  : {pred_csv}")
            logger.info(f"  Duration    : {duration:.1f}s")
            return artifact

        except Exception as e:
            self.result.stages_failed.append(stage)
            raise PipelineException(e, sys, stage=stage) from e

    # ────────────────────────────────────────────────────────────────────
    # STAGE 5: Evaluation  (predictions → reports)
    # ────────────────────────────────────────────────────────────────────
    def start_evaluation(
        self,
        inference_artifact: InferenceArtifact,
        config: Optional[EvaluationConfig] = None,
    ) -> EvaluationArtifact:
        """Analyze prediction distribution, confidence, ECE."""
        stage = "evaluation"
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE 5: EVALUATION")
        logger.info(f"{'='*60}")
        t0 = time.time()

        try:
            from evaluate_predictions import (
                EvaluationPipeline as _EvalPipeline,
                EvaluationConfig as _EvalCfg,
            )

            cfg = config or EvaluationConfig()
            eval_cfg = _EvalCfg()

            if cfg.output_dir:
                eval_cfg.output_dir = cfg.output_dir
            else:
                eval_cfg.output_dir = self.cfg.outputs_dir / "evaluation"

            pred_csv = cfg.predictions_csv or inference_artifact.predictions_csv_path

            pipe = _EvalPipeline(eval_cfg)
            result = pipe.run(predictions_csv=pred_csv)

            artifact = EvaluationArtifact(
                distribution_summary=result.get("results", {}).get("distribution", {}),
                confidence_summary=result.get("results", {}).get("confidence", {}),
                has_labels=False,
            )
            if result.get("results", {}).get("output_files"):
                files = result["results"]["output_files"]
                artifact.report_json_path = Path(files.get("json", ""))
                artifact.report_text_path = Path(files.get("txt", ""))

            self.result.evaluation = artifact
            self.result.stages_completed.append(stage)
            logger.info(f"  Duration : {time.time()-t0:.1f}s")
            return artifact

        except Exception as e:
            self.result.stages_failed.append(stage)
            raise PipelineException(e, sys, stage=stage) from e

    # ────────────────────────────────────────────────────────────────────
    # STAGE 6: Post-Inference Monitoring  (3-layer)
    # ────────────────────────────────────────────────────────────────────
    def start_monitoring(
        self,
        inference_artifact: InferenceArtifact,
        preprocessing_artifact: Optional[PreprocessingArtifact] = None,
        config: Optional[MonitoringConfig] = None,
    ) -> MonitoringArtifact:
        """Run 3-layer monitoring: confidence, temporal, drift."""
        stage = "monitoring"
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE 6: POST-INFERENCE MONITORING")
        logger.info(f"{'='*60}")
        t0 = time.time()

        try:
            # post_inference_monitoring lives in scripts/
            scripts_dir = self.cfg.project_root / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from post_inference_monitoring import PostInferenceMonitor

            cfg = config or MonitoringConfig()

            monitor = PostInferenceMonitor()

            pred_csv = cfg.predictions_csv or inference_artifact.predictions_csv_path
            prod_npy = None
            baseline_json = None

            if preprocessing_artifact:
                prod_npy = cfg.production_data_npy or preprocessing_artifact.production_X_path
            baseline_json = cfg.baseline_stats_json or (
                self.cfg.data_prepared_dir / "baseline_stats.json"
            )

            report = monitor.run(
                predictions_path=Path(pred_csv),
                production_data_path=Path(prod_npy) if prod_npy else None,
                baseline_path=Path(baseline_json) if baseline_json and Path(baseline_json).exists() else None,
                output_dir=cfg.output_dir or self.cfg.outputs_dir / "monitoring",
            )

            # Extract overall status from the report object
            overall = "UNKNOWN"
            if hasattr(report, "overall_status"):
                overall = report.overall_status
            elif isinstance(report, dict):
                overall = report.get("overall_status", "UNKNOWN")

            artifact = MonitoringArtifact(
                monitoring_report=report if isinstance(report, dict) else vars(report) if hasattr(report, "__dict__") else {},
                overall_status=overall,
            )

            self.result.monitoring = artifact
            self.result.stages_completed.append(stage)
            logger.info(f"  Status   : {overall}")
            logger.info(f"  Duration : {time.time()-t0:.1f}s")
            return artifact

        except Exception as e:
            self.result.stages_failed.append(stage)
            raise PipelineException(e, sys, stage=stage) from e

    # ────────────────────────────────────────────────────────────────────
    # STAGE 7: Trigger Evaluation  (should we retrain?)
    # ────────────────────────────────────────────────────────────────────
    def start_trigger_evaluation(
        self,
        monitoring_artifact: MonitoringArtifact,
        config: Optional[TriggerConfig] = None,
    ) -> TriggerArtifact:
        """Evaluate retraining triggers using 2-of-3 voting."""
        stage = "trigger_evaluation"
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE 7: TRIGGER EVALUATION")
        logger.info(f"{'='*60}")
        t0 = time.time()

        try:
            from trigger_policy import TriggerPolicyEngine

            engine = TriggerPolicyEngine()
            decision = engine.evaluate(monitoring_artifact.monitoring_report)

            # Normalize decision to dict if it's a dataclass
            if hasattr(decision, "__dict__") and not isinstance(decision, dict):
                dec = vars(decision)
            else:
                dec = decision if isinstance(decision, dict) else {}

            artifact = TriggerArtifact(
                should_retrain=dec.get("should_trigger", dec.get("should_retrain", False)),
                action=str(dec.get("action", "NONE")),
                alert_level=str(dec.get("alert_level", "INFO")),
                reasons=dec.get("reasons", dec.get("reason", [])),
                cooldown_active=dec.get("cooldown_active", False),
            )
            # Ensure reasons is a list
            if isinstance(artifact.reasons, str):
                artifact.reasons = [artifact.reasons]

            self.result.trigger = artifact
            self.result.stages_completed.append(stage)

            logger.info(f"  Retrain  : {'YES' if artifact.should_retrain else 'NO'}")
            logger.info(f"  Action   : {artifact.action}")
            logger.info(f"  Alert    : {artifact.alert_level}")
            if artifact.reasons:
                for r in artifact.reasons:
                    logger.info(f"  Reason   : {r}")
            logger.info(f"  Duration : {time.time()-t0:.1f}s")
            return artifact

        except Exception as e:
            self.result.stages_failed.append(stage)
            raise PipelineException(e, sys, stage=stage) from e

    # ════════════════════════════════════════════════════════════════════
    # RUN PIPELINE  (the main method)
    # ════════════════════════════════════════════════════════════════════
    def run(
        self,
        stages: Optional[List[str]] = None,
        skip_ingestion: bool = False,
        skip_validation: bool = False,
        continue_on_failure: bool = False,
    ) -> PipelineResult:
        """
        Execute the full production pipeline (or selected stages).

        Args:
            stages: List of stage names to run, e.g. ["preprocess", "infer"].
                    If None, runs all stages in order.
            skip_ingestion: If True, skip data ingestion and use existing
                           sensor_fused_50Hz.csv from data/processed/.
            skip_validation: If True, skip data validation.
            continue_on_failure: If True, log errors and continue to next stage
                                instead of aborting.

        Returns:
            PipelineResult with all artifacts and status.
        """
        all_stages = [
            "ingestion",
            "validation",
            "preprocessing",
            "inference",
            "evaluation",
            "monitoring",
            "trigger",
        ]

        if stages:
            run_stages = [s for s in all_stages if s in stages]
        else:
            run_stages = list(all_stages)

        if skip_ingestion and "ingestion" in run_stages:
            run_stages.remove("ingestion")
        if skip_validation and "validation" in run_stages:
            run_stages.remove("validation")

        logger.info(f"\nStages to run: {run_stages}\n")

        # Artifacts flowing between stages
        ingestion_art: Optional[DataIngestionArtifact] = None
        validation_art: Optional[DataValidationArtifact] = None
        preprocess_art: Optional[PreprocessingArtifact] = None
        inference_art: Optional[InferenceArtifact] = None
        evaluation_art: Optional[EvaluationArtifact] = None
        monitoring_art: Optional[MonitoringArtifact] = None
        trigger_art: Optional[TriggerArtifact] = None

        def _run_stage(stage_name: str, fn, *args, **kwargs):
            """Helper to run a stage with error handling."""
            try:
                return fn(*args, **kwargs)
            except PipelineException:
                if not continue_on_failure:
                    raise
                logger.error(f"Stage '{stage_name}' failed — continuing", exc_info=True)
                # stage already added to stages_failed inside start_* method
                return None
            except Exception as exc:
                if not continue_on_failure:
                    raise PipelineException(exc, sys, stage=stage_name) from exc
                logger.error(f"Stage '{stage_name}' failed — continuing", exc_info=True)
                if stage_name not in self.result.stages_failed:
                    self.result.stages_failed.append(stage_name)
                return None

        # ── Stage 1: Ingestion ─────────────────────────────────────────
        if "ingestion" in run_stages:
            ingestion_art = _run_stage("ingestion", self.start_data_ingestion)
        else:
            # Create a synthetic artifact from existing processed CSV
            fused_csv = self.cfg.data_processed_dir / "sensor_fused_50Hz.csv"
            if fused_csv.exists():
                logger.info(f"Skipping ingestion — using existing {fused_csv}")
                ingestion_art = DataIngestionArtifact(
                    fused_csv_path=fused_csv,
                    n_rows=-1,
                    n_columns=-1,
                    sampling_hz=50,
                    ingestion_timestamp="existing",
                )
                self.result.stages_skipped.append("ingestion")
            else:
                logger.warning("No existing fused CSV found — ingestion required!")

        # ── Stage 2: Validation ────────────────────────────────────────
        if "validation" in run_stages and ingestion_art:
            validation_art = _run_stage(
                "validation", self.start_data_validation, ingestion_art
            )

        # ── Stage 3: Preprocessing ────────────────────────────────────
        if "preprocessing" in run_stages and ingestion_art:
            preprocess_art = _run_stage(
                "preprocessing", self.start_preprocessing, ingestion_art
            )
        else:
            # Use existing production_X.npy
            prod_x = self.cfg.data_prepared_dir / "production_X.npy"
            meta = self.cfg.data_prepared_dir / "production_metadata.json"
            if prod_x.exists():
                import numpy as np
                X = np.load(prod_x)
                logger.info(f"Skipping preprocessing — using existing {prod_x} ({X.shape})")
                preprocess_art = PreprocessingArtifact(
                    production_X_path=prod_x,
                    metadata_path=meta,
                    n_windows=X.shape[0],
                    window_size=X.shape[1] if len(X.shape) > 1 else 0,
                    unit_conversion_applied=False,
                    preprocessing_timestamp="existing",
                )
                self.result.stages_skipped.append("preprocessing")

        # ── Stage 4: Inference ─────────────────────────────────────────
        if "inference" in run_stages and preprocess_art:
            inference_art = _run_stage(
                "inference", self.start_inference, preprocess_art
            )

        # ── Stage 5: Evaluation ────────────────────────────────────────
        if "evaluation" in run_stages and inference_art:
            evaluation_art = _run_stage(
                "evaluation", self.start_evaluation, inference_art
            )

        # ── Stage 6: Monitoring ────────────────────────────────────────
        if "monitoring" in run_stages and inference_art:
            monitoring_art = _run_stage(
                "monitoring",
                self.start_monitoring,
                inference_art,
                preprocess_art,
            )

        # ── Stage 7: Trigger Evaluation ────────────────────────────────
        if "trigger" in run_stages and monitoring_art:
            trigger_art = _run_stage(
                "trigger", self.start_trigger_evaluation, monitoring_art
            )

        # ── Finalize ───────────────────────────────────────────────────
        self.result.end_time = datetime.now().isoformat()
        if self.result.stages_failed:
            self.result.overall_status = "FAILED"
        elif not self.result.stages_completed:
            self.result.overall_status = "NO_STAGES_RAN"
        else:
            self.result.overall_status = "SUCCESS"

        self._print_summary()
        self._save_result()

        return self.result

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────
    def _print_summary(self):
        """Print a human-readable pipeline summary."""
        r = self.result
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Run ID     : {r.run_id}")
        logger.info(f"  Status     : {r.overall_status}")
        logger.info(f"  Started    : {r.start_time}")
        logger.info(f"  Finished   : {r.end_time}")
        logger.info(f"  Completed  : {r.stages_completed}")
        logger.info(f"  Skipped    : {r.stages_skipped}")
        logger.info(f"  Failed     : {r.stages_failed}")

        if r.inference and r.inference.n_predictions:
            logger.info(f"  Predictions: {r.inference.n_predictions}")
        if r.trigger:
            logger.info(f"  Retrain?   : {'YES' if r.trigger.should_retrain else 'NO'}")
        logger.info("=" * 70)

    def _save_result(self):
        """Save pipeline result as JSON for traceability."""
        out_dir = self.cfg.logs_dir / "pipeline"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"pipeline_result_{self.result.run_id}.json"

        # Convert to serializable dict
        def _to_dict(obj):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_dict(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _to_dict(v) for k, v in vars(obj).items()}
            return obj

        try:
            with open(out_file, "w") as f:
                json.dump(_to_dict(self.result), f, indent=2, default=str)
            logger.info(f"  Result saved: {out_file}")
        except Exception as e:
            logger.warning(f"  Could not save result JSON: {e}")
