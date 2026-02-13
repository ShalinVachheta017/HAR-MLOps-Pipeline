"""
Artifact entities for HAR MLOps Production Pipeline.

Each pipeline component produces an artifact dataclass that is passed to the
next component.  This makes the data flow explicit and traceable.

Stages 1-7  : inference-cycle artifacts
Stages 8-10 : retraining-cycle artifacts
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path


# ============================================================================
# Stage 1 – Data Ingestion
# ============================================================================

@dataclass
class DataIngestionArtifact:
    """Output of sensor-data ingestion (Excel/CSV → fused CSV)."""
    fused_csv_path: Path
    n_rows: int
    n_columns: int
    sampling_hz: int
    ingestion_timestamp: str
    source_type: str = "excel"             # "excel" or "csv"


# ============================================================================
# Stage 2 – Data Validation
# ============================================================================

@dataclass
class DataValidationArtifact:
    """Output of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)


# ============================================================================
# Stage 3 – Data Transformation
# ============================================================================

@dataclass
class DataTransformationArtifact:
    """Output of preprocessing (CSV → windowed .npy)."""
    production_X_path: Path
    metadata_path: Path
    n_windows: int
    window_size: int
    unit_conversion_applied: bool
    preprocessing_timestamp: str


# ============================================================================
# Stage 4 – Model Inference
# ============================================================================

@dataclass
class ModelInferenceArtifact:
    """Output of model inference."""
    predictions_csv_path: Path
    predictions_npy_path: Path
    probabilities_npy_path: Optional[Path] = None
    n_predictions: int = 0
    inference_time_seconds: float = 0.0
    model_version: str = ""


# ============================================================================
# Stage 5 – Model Evaluation
# ============================================================================

@dataclass
class ModelEvaluationArtifact:
    """Output of prediction evaluation."""
    report_json_path: Optional[Path] = None
    report_text_path: Optional[Path] = None
    distribution_summary: Dict = field(default_factory=dict)
    confidence_summary: Dict = field(default_factory=dict)
    has_labels: bool = False
    classification_metrics: Optional[Dict] = None


# ============================================================================
# Stage 6 – Post-Inference Monitoring
# ============================================================================

@dataclass
class PostInferenceMonitoringArtifact:
    """Output of post-inference monitoring (3-layer)."""
    monitoring_report: Dict = field(default_factory=dict)
    overall_status: str = "UNKNOWN"        # HEALTHY / WARNING / CRITICAL
    layer1_confidence: Dict = field(default_factory=dict)
    layer2_temporal: Dict = field(default_factory=dict)
    layer3_drift: Dict = field(default_factory=dict)
    report_path: Optional[Path] = None


# ============================================================================
# Stage 7 – Trigger Evaluation
# ============================================================================

@dataclass
class TriggerEvaluationArtifact:
    """Output of retraining-trigger evaluation."""
    should_retrain: bool = False
    action: str = "NONE"                   # NONE / MONITOR / QUEUE_RETRAIN / TRIGGER_RETRAIN / ROLLBACK
    alert_level: str = "INFO"              # INFO / WARNING / CRITICAL
    reasons: List[str] = field(default_factory=list)
    cooldown_active: bool = False


# ============================================================================
# Stage 8 – Model Retraining
# ============================================================================

@dataclass
class ModelRetrainingArtifact:
    """Output of model retraining (standard or domain-adapted)."""
    retrained_model_path: Optional[Path] = None
    scaler_config_path: Optional[Path] = None     # new scaler params
    training_report: Dict = field(default_factory=dict)
    adaptation_method: str = "none"               # adabn / pseudo_label / supervised / none
    metrics: Dict[str, float] = field(default_factory=dict)
    n_source_samples: int = 0
    n_target_samples: int = 0
    retraining_timestamp: str = ""


# ============================================================================
# Stage 9 – Model Registration
# ============================================================================

@dataclass
class ModelRegistrationArtifact:
    """Output of model registration and validation."""
    registered_version: str = ""
    is_deployed: bool = False
    is_better_than_current: bool = False
    proxy_metrics: Dict[str, float] = field(default_factory=dict)
    previous_version: Optional[str] = None
    registry_path: Optional[Path] = None


# ============================================================================
# Stage 10 – Baseline Update
# ============================================================================

@dataclass
class BaselineUpdateArtifact:
    """Output of baseline rebuild."""
    baseline_path: Optional[Path] = None
    normalized_baseline_path: Optional[Path] = None
    n_channels: int = 0
    stats_summary: Dict = field(default_factory=dict)
    update_timestamp: str = ""


# ============================================================================
# Stage 11 – Calibration & Uncertainty Quantification
# ============================================================================

@dataclass
class CalibrationUncertaintyArtifact:
    """Output of calibration and uncertainty quantification."""
    temperature: float = 1.0
    temperature_path: Optional[Path] = None
    ece: float = 0.0
    mce: float = 0.0
    brier_score: float = 0.0
    overconfidence_gap: float = 0.0
    reliability_diagram_path: Optional[Path] = None
    mc_dropout_run: bool = False
    mean_predictive_entropy: float = 0.0
    mean_mutual_information: float = 0.0
    calibration_report: Dict = field(default_factory=dict)
    calibration_warnings: List[str] = field(default_factory=list)


# ============================================================================
# Stage 12 – Wasserstein Drift Detection
# ============================================================================

@dataclass
class WassersteinDriftArtifact:
    """Output of Wasserstein-based drift detection."""
    overall_status: str = "UNKNOWN"
    mean_wasserstein: float = 0.0
    max_wasserstein: float = 0.0
    n_channels_warn: int = 0
    n_channels_critical: int = 0
    per_channel: Dict = field(default_factory=dict)
    change_points: List[int] = field(default_factory=list)
    drift_trend: str = "UNKNOWN"
    integrated_report: Dict = field(default_factory=dict)
    report_path: Optional[Path] = None


# ============================================================================
# Stage 13 – Curriculum Pseudo-Labeling
# ============================================================================

@dataclass
class CurriculumPseudoLabelingArtifact:
    """Output of curriculum pseudo-labeling training."""
    retrained_model_path: Optional[Path] = None
    iterations_completed: int = 0
    total_pseudo_labeled: int = 0
    best_val_accuracy: float = 0.0
    final_mean_confidence: float = 0.0
    iteration_logs: List[Dict] = field(default_factory=list)
    ewc_used: bool = False
    training_report: Dict = field(default_factory=dict)


# ============================================================================
# Stage 14 – Sensor Placement Robustness
# ============================================================================

@dataclass
class SensorPlacementArtifact:
    """Output of sensor placement analysis."""
    detected_hand: str = "UNKNOWN"
    detection_confidence: float = 0.0
    augmented_data_path: Optional[Path] = None
    n_original_samples: int = 0
    n_augmented_samples: int = 0
    per_hand_report: Dict = field(default_factory=dict)
    hand_features: Dict = field(default_factory=dict)


# ============================================================================
# Pipeline Result (aggregated)
# ============================================================================

@dataclass
class PipelineResult:
    """Aggregated result of the full pipeline run."""
    run_id: str = ""
    start_time: str = ""
    end_time: str = ""
    overall_status: str = "UNKNOWN"
    stages_completed: List[str] = field(default_factory=list)
    stages_skipped: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)

    # Artifacts from each stage (None = stage did not run)
    ingestion: Optional[DataIngestionArtifact] = None
    validation: Optional[DataValidationArtifact] = None
    transformation: Optional[DataTransformationArtifact] = None
    inference: Optional[ModelInferenceArtifact] = None
    evaluation: Optional[ModelEvaluationArtifact] = None
    monitoring: Optional[PostInferenceMonitoringArtifact] = None
    trigger: Optional[TriggerEvaluationArtifact] = None
    retraining: Optional[ModelRetrainingArtifact] = None
    registration: Optional[ModelRegistrationArtifact] = None
    baseline_update: Optional[BaselineUpdateArtifact] = None
    calibration: Optional["CalibrationUncertaintyArtifact"] = None
    wasserstein_drift: Optional["WassersteinDriftArtifact"] = None
    curriculum_pseudo_labeling: Optional["CurriculumPseudoLabelingArtifact"] = None
    sensor_placement: Optional["SensorPlacementArtifact"] = None
