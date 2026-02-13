"""
Component 11 – Calibration & Uncertainty Quantification

Wraps:  src/calibration.py  →  TemperatureScaler, CalibrationEvaluator,
        MCDropoutEstimator, UnlabeledCalibrationAnalyzer
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.entity.config_entity import CalibrationUncertaintyConfig, PipelineConfig
from src.entity.artifact_entity import (
    ModelInferenceArtifact,
    CalibrationUncertaintyArtifact,
)

logger = logging.getLogger(__name__)


class CalibrationUncertainty:
    """Post-hoc calibration and uncertainty quantification for predictions."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        config: CalibrationUncertaintyConfig,
        inference_artifact: ModelInferenceArtifact,
    ):
        self.pipeline_config = pipeline_config
        self.config = config
        self.inference_artifact = inference_artifact

    # ------------------------------------------------------------------ #
    def initiate_calibration(self) -> CalibrationUncertaintyArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 11 — Calibration & Uncertainty Quantification")
        logger.info("=" * 60)

        from src.calibration import (
            CalibrationConfig as _CalConfig,
            TemperatureScaler,
            CalibrationEvaluator,
            UnlabeledCalibrationAnalyzer,
        )

        # Output directory
        output_dir = Path(
            self.config.output_dir
            or self.pipeline_config.outputs_dir / "calibration"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load probabilities
        probs_path = self.inference_artifact.probabilities_npy_path
        if probs_path and Path(probs_path).exists():
            probs = np.load(probs_path)
        else:
            logger.warning("No probabilities found — using predictions CSV.")
            import pandas as pd
            csv_path = self.inference_artifact.predictions_csv_path
            df = pd.read_csv(csv_path)
            if "confidence" in df.columns:
                # Reconstruct rough probs from confidence + predicted_label
                n_classes = 11
                probs = np.full((len(df), n_classes), (1 - df["confidence"].values[:, None]) / (n_classes - 1))
                for i, row in df.iterrows():
                    probs[i, int(row.get("predicted_label", 0))] = row["confidence"]
            else:
                return CalibrationUncertaintyArtifact(
                    calibration_warnings=["No probability data available for calibration."]
                )

        # --- Temperature Scaling ---
        cal_cfg = _CalConfig(
            initial_temperature=self.config.initial_temperature,
            lr=self.config.temp_lr,
            max_iter=self.config.temp_max_iter,
            n_bins=self.config.n_bins,
        )
        scaler = TemperatureScaler(cal_cfg)

        # Load or fit temperature
        temp_path = Path(
            self.config.temperature_path
            or output_dir / "temperature.json"
        )
        if temp_path.exists():
            scaler.load(temp_path)
            logger.info("Loaded existing temperature: T=%.4f", scaler.temperature)
        else:
            # Without validation labels, use default T or fit if labels available
            logger.info(
                "No saved temperature — using initial T=%.4f. "
                "Fit on validation data for optimal calibration.",
                scaler.temperature,
            )

        # Apply temperature scaling
        # Convert probs back to logits, scale, then re-softmax
        logits = np.log(probs + 1e-10)
        calibrated_probs = scaler.transform(logits)

        # Save calibrated probabilities
        cal_probs_path = output_dir / "calibrated_probabilities.npy"
        np.save(cal_probs_path, calibrated_probs)

        # Save temperature
        scaler.save(temp_path)

        # --- Unlabeled calibration analysis ---
        analyzer = UnlabeledCalibrationAnalyzer()
        analysis = analyzer.analyze(calibrated_probs, scaler.temperature)

        # --- Calibration report ---
        evaluator = CalibrationEvaluator(n_bins=self.config.n_bins)

        # Save report
        import json
        report_path = output_dir / "calibration_report.json"
        report_data = {
            "temperature": scaler.temperature,
            "analysis": analysis,
        }
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return CalibrationUncertaintyArtifact(
            temperature=scaler.temperature,
            temperature_path=temp_path,
            overconfidence_gap=analysis.get("overconfidence_ratio", 0.0),
            mean_predictive_entropy=analysis.get("mean_entropy", 0.0),
            calibration_report=analysis,
            calibration_warnings=analysis.get("calibration_warnings", []),
            reliability_diagram_path=None,
        )
