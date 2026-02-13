"""
Configuration entities for HAR MLOps Production Pipeline.

Each pipeline stage gets a dataclass config.  The ProductionPipeline reads
these at init, and each Component receives only the config it needs.

Inspired by: vikashishere/YT-MLops-Proj1 (see docs/thesis/production refrencxe/)

Stages:
     1  Data Ingestion         (Excel/CSV → fused CSV)
     2  Data Validation        (schema + range checks)
     3  Data Transformation    (CSV → windowed .npy)
     4  Model Inference        (.npy + model → predictions)
     5  Model Evaluation       (confidence / distribution / ECE)
     6  Post-Inference Monitoring (3-layer: confidence, temporal, drift)
     7  Trigger Evaluation     (retraining decision)
     8  Model Retraining       (unsupervised / supervised, AdaBN)
     9  Model Registration     (version, deploy, rollback)
    10  Baseline Update        (rebuild drift baselines)
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import os

TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# Master Pipeline Config
# ============================================================================

@dataclass
class PipelineConfig:
    """Top-level pipeline settings shared by all stages."""
    pipeline_name: str = "har_mental_health_pipeline"
    artifact_dir: str = os.path.join("artifacts", TIMESTAMP)
    timestamp: str = TIMESTAMP

    # Paths — sensible defaults, override via run_pipeline.py CLI
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_raw_dir: Path = field(default=None)
    data_processed_dir: Path = field(default=None)
    data_prepared_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    models_pretrained_dir: Path = field(default=None)
    outputs_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    scripts_dir: Path = field(default=None)

    def __post_init__(self):
        if self.data_raw_dir is None:
            self.data_raw_dir = self.project_root / "data" / "raw"
        if self.data_processed_dir is None:
            self.data_processed_dir = self.project_root / "data" / "processed"
        if self.data_prepared_dir is None:
            self.data_prepared_dir = self.project_root / "data" / "prepared"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models"
        if self.models_pretrained_dir is None:
            self.models_pretrained_dir = self.project_root / "models" / "pretrained"
        if self.outputs_dir is None:
            self.outputs_dir = self.project_root / "outputs"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        if self.scripts_dir is None:
            self.scripts_dir = self.project_root / "scripts"


# ============================================================================
# Stage 1 – Data Ingestion
# ============================================================================

@dataclass
class DataIngestionConfig:
    """Configuration for raw-sensor-data ingestion (Excel/CSV → fused CSV)."""
    accel_file: Optional[Path] = None          # auto-detect if None
    gyro_file: Optional[Path] = None           # auto-detect if None
    input_csv: Optional[Path] = None           # direct CSV (your own recording)
    target_hz: int = 50
    merge_tolerance_ms: int = 1
    output_dir: Optional[Path] = None


# ============================================================================
# Stage 2 – Data Validation
# ============================================================================

@dataclass
class DataValidationConfig:
    """Configuration for data validation."""
    sensor_columns: List[str] = field(
        default_factory=lambda: ["Ax_w", "Ay_w", "Az_w", "Gx_w", "Gy_w", "Gz_w"]
    )
    expected_frequency_hz: float = 50.0
    max_acceleration_ms2: float = 50.0
    max_gyroscope_dps: float = 500.0
    max_missing_ratio: float = 0.05


# ============================================================================
# Stage 3 – Data Transformation (Preprocessing)
# ============================================================================

@dataclass
class DataTransformationConfig:
    """Configuration for preprocessing (CSV → windowed .npy arrays)."""
    input_csv: Optional[Path] = None
    enable_gravity_removal: bool = False
    enable_calibration: bool = False
    window_size: int = 200                 # 4 seconds at 50 Hz
    overlap: float = 0.5                   # 50 % overlap
    sensors: List[str] = field(
        default_factory=lambda: ["Ax_w", "Ay_w", "Az_w", "Gx_w", "Gy_w", "Gz_w"]
    )
    output_dir: Optional[Path] = None


# ============================================================================
# Stage 4 – Model Inference
# ============================================================================

@dataclass
class ModelInferenceConfig:
    """Configuration for model inference."""
    model_path: Optional[Path] = None      # defaults to pretrained
    input_npy: Optional[Path] = None       # defaults to prepared/production_X.npy
    batch_size: int = 32
    confidence_threshold: float = 0.50
    mode: str = "batch"                    # "batch" or "realtime"
    output_dir: Optional[Path] = None


# ============================================================================
# Stage 5 – Model Evaluation
# ============================================================================

@dataclass
class ModelEvaluationConfig:
    """Configuration for prediction evaluation."""
    predictions_csv: Optional[Path] = None
    labels_path: Optional[Path] = None     # optional — unlabeled is fine
    output_dir: Optional[Path] = None
    confidence_bins: int = 10


# ============================================================================
# Stage 6 – Post-Inference Monitoring
# ============================================================================

@dataclass
class PostInferenceMonitoringConfig:
    """Configuration for post-inference monitoring (3-layer)."""
    predictions_csv: Optional[Path] = None
    production_data_npy: Optional[Path] = None
    baseline_stats_json: Optional[Path] = None
    model_path: Optional[Path] = None
    output_dir: Optional[Path] = None


# ============================================================================
# Stage 7 – Trigger Evaluation
# ============================================================================

@dataclass
class TriggerEvaluationConfig:
    """Configuration for retraining-trigger evaluation."""
    confidence_warn: float = 0.65
    confidence_critical: float = 0.50
    drift_psi_warn: float = 0.1
    drift_psi_critical: float = 0.2
    temporal_flip_warn: float = 0.35
    temporal_flip_critical: float = 0.50
    cooldown_hours: int = 24
    state_dir: Optional[Path] = None


# ============================================================================
# Stage 8 – Model Retraining
# ============================================================================

@dataclass
class ModelRetrainingConfig:
    """Configuration for model retraining (including domain adaptation)."""
    # Data paths
    source_data_path: Optional[Path] = None        # labeled training CSV
    target_data_npy: Optional[Path] = None          # unlabeled production .npy
    labels_path: Optional[Path] = None              # user-provided labels (optional)

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    n_folds: int = 5
    skip_cv: bool = False

    # Domain adaptation
    enable_adaptation: bool = False
    adaptation_method: str = "adabn"               # adabn, pseudo_label, mmd, dann
    adabn_n_batches: int = 10                      # how many batches for AdaBN stat update

    # Output
    output_dir: Optional[Path] = None
    experiment_name: str = "har-retraining"
    run_name: Optional[str] = None


# ============================================================================
# Stage 9 – Model Registration
# ============================================================================

@dataclass
class ModelRegistrationConfig:
    """Configuration for model versioning and deployment."""
    registry_dir: Optional[Path] = None
    model_path: Optional[Path] = None             # new model to register
    version: Optional[str] = None                 # auto-incremented if None
    auto_deploy: bool = False                     # deploy immediately if better
    proxy_validation: bool = True                 # validate before deployment


# ============================================================================
# Stage 10 – Baseline Update
# ============================================================================

@dataclass
class BaselineUpdateConfig:
    """Configuration for rebuilding drift baselines after retraining."""
    training_data_path: Optional[Path] = None     # CSV with labeled data
    scaler_config_path: Optional[Path] = None     # scaler params for normalized baseline
    output_baseline_path: Optional[Path] = None   # where to save baseline JSON
    output_normalized_path: Optional[Path] = None # where to save normalized baseline
    rebuild_embeddings: bool = False
    model_path: Optional[Path] = None             # for embedding baseline


# ============================================================================
# Stage 11 – Calibration & Uncertainty Quantification
# ============================================================================

@dataclass
class CalibrationUncertaintyConfig:
    """Configuration for post-hoc calibration and MC Dropout UQ."""
    # Temperature scaling
    initial_temperature: float = 1.5
    temp_lr: float = 0.01
    temp_max_iter: int = 100

    # MC Dropout
    mc_forward_passes: int = 30
    mc_dropout_rate: float = 0.2

    # Thresholds
    confidence_warn_threshold: float = 0.65
    entropy_warn_threshold: float = 1.5
    ece_warn_threshold: float = 0.10

    # Evaluation bins
    n_bins: int = 15

    # Output
    save_reliability_diagram: bool = True
    temperature_path: Optional[Path] = None
    output_dir: Optional[Path] = None


# ============================================================================
# Stage 12 – Wasserstein Drift Detection
# ============================================================================

@dataclass
class WassersteinDriftConfig:
    """Configuration for Wasserstein-based distribution drift detection."""
    warn_threshold: float = 0.3
    critical_threshold: float = 0.5
    min_drifted_channels_warn: int = 2
    min_drifted_channels_critical: int = 4

    # Change-point detection
    cpd_window_size: int = 50
    cpd_threshold: float = 2.0

    # Multi-resolution
    enable_multi_resolution: bool = True

    sensor_columns: List[str] = field(
        default_factory=lambda: ["Ax_w", "Ay_w", "Az_w", "Gx_w", "Gy_w", "Gz_w"]
    )
    baseline_data_path: Optional[Path] = None
    output_dir: Optional[Path] = None


# ============================================================================
# Stage 13 – Curriculum Pseudo-Labeling
# ============================================================================

@dataclass
class CurriculumPseudoLabelingConfig:
    """Configuration for curriculum-style self-training."""
    initial_confidence_threshold: float = 0.95
    final_confidence_threshold: float = 0.80
    n_iterations: int = 5
    threshold_decay: str = "linear"        # linear or exponential

    max_samples_per_class: int = 20
    min_samples_per_class: int = 3

    use_teacher_student: bool = True
    ema_decay: float = 0.999

    # EWC regularization
    use_ewc: bool = True
    ewc_lambda: float = 1000.0
    ewc_n_samples: int = 200

    epochs_per_iteration: int = 10
    batch_size: int = 64
    learning_rate: float = 0.0005

    source_data_path: Optional[Path] = None
    unlabeled_data_path: Optional[Path] = None
    output_dir: Optional[Path] = None


# ============================================================================
# Stage 14 – Sensor Placement Robustness
# ============================================================================

@dataclass
class SensorPlacementConfig:
    """Configuration for sensor placement robustness (hand detection, mirroring)."""
    mirror_axes: List[int] = field(default_factory=lambda: [1, 2, 4, 5])
    mirror_probability: float = 0.5
    dominant_accel_threshold: float = 1.2

    sensor_columns: List[str] = field(
        default_factory=lambda: ["Ax_w", "Ay_w", "Az_w", "Gx_w", "Gy_w", "Gz_w"]
    )
    accel_indices: List[int] = field(default_factory=lambda: [0, 1, 2])
    gyro_indices: List[int] = field(default_factory=lambda: [3, 4, 5])
    output_dir: Optional[Path] = None
