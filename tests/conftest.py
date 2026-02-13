"""
Pytest Configuration and Fixtures
=================================

Shared fixtures for the HAR MLOps test suite.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil


# ============================================================================
# PATH FIXTURES
# ============================================================================

@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


# ============================================================================
# DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate 6-axis IMU data (Ax, Ay, Az, Gx, Gy, Gz)
    # Accelerometer in m/s² (typical range: -20 to 20)
    # Gyroscope in deg/s (typical range: -250 to 250)
    data = {
        'timestamp': pd.date_range('2026-01-30', periods=n_samples, freq='20ms'),
        'Ax_w': np.random.normal(0, 2, n_samples),  # Accelerometer X
        'Ay_w': np.random.normal(0, 2, n_samples),  # Accelerometer Y  
        'Az_w': np.random.normal(9.8, 2, n_samples),  # Accelerometer Z (gravity)
        'Gx_w': np.random.normal(0, 50, n_samples),  # Gyroscope X
        'Gy_w': np.random.normal(0, 50, n_samples),  # Gyroscope Y
        'Gz_w': np.random.normal(0, 50, n_samples),  # Gyroscope Z
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_labeled_data():
    """Generate sample labeled training data."""
    np.random.seed(42)
    n_samples = 5000
    
    activities = [
        'sitting', 'standing', 'walking', 'hand_tapping',
        'nail_biting', 'hair_pulling', 'smoking'
    ]
    
    # Generate data for each activity
    rows = []
    for i in range(n_samples):
        activity = activities[i % len(activities)]
        
        # Activity-specific patterns
        if activity in ['sitting', 'standing']:
            ax = np.random.normal(0, 0.5)
            ay = np.random.normal(0, 0.5)
            az = np.random.normal(9.8, 0.3)
            gx = np.random.normal(0, 10)
            gy = np.random.normal(0, 10)
            gz = np.random.normal(0, 10)
        else:
            ax = np.random.normal(0, 3)
            ay = np.random.normal(0, 3)
            az = np.random.normal(9.8, 2)
            gx = np.random.normal(0, 100)
            gy = np.random.normal(0, 100)
            gz = np.random.normal(0, 100)
        
        rows.append({
            'User': f'user_{i % 5}',
            'activity': activity,
            'Ax_w': ax,
            'Ay_w': ay,
            'Az_w': az,
            'Gx_w': gx,
            'Gy_w': gy,
            'Gz_w': gz,
        })
    
    return pd.DataFrame(rows)


@pytest.fixture
def sample_windows():
    """Generate sample windowed data for inference testing."""
    np.random.seed(42)
    n_windows = 50
    window_size = 200
    n_sensors = 6
    
    # Random windows
    X = np.random.randn(n_windows, window_size, n_sensors)
    
    # Normalize to typical sensor ranges
    X[:, :, :3] *= 2  # Accelerometer
    X[:, :, 3:] *= 50  # Gyroscope
    
    return X


@pytest.fixture
def sample_predictions():
    """Generate sample model predictions for monitoring tests."""
    np.random.seed(42)
    n_predictions = 100
    n_classes = 11
    
    # Generate probability distributions
    probs = np.random.dirichlet(np.ones(n_classes) * 2, size=n_predictions)
    
    # Make some predictions more confident
    for i in range(0, n_predictions, 3):
        dominant_class = np.random.randint(0, n_classes)
        probs[i] = np.zeros(n_classes)
        probs[i, dominant_class] = 0.95
        probs[i] += np.random.uniform(0, 0.01, n_classes)
        probs[i] /= probs[i].sum()
    
    predictions = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    
    return {
        'probabilities': probs,
        'predictions': predictions,
        'confidence': confidence
    }


# ============================================================================
# BASELINE FIXTURES
# ============================================================================

@pytest.fixture
def sample_baseline_stats():
    """Generate sample baseline statistics for drift detection."""
    return {
        'Ax_w': {'mean': 0.0, 'std': 2.0, 'min': -10, 'max': 10},
        'Ay_w': {'mean': 0.0, 'std': 2.0, 'min': -10, 'max': 10},
        'Az_w': {'mean': 9.8, 'std': 2.0, 'min': 0, 'max': 20},
        'Gx_w': {'mean': 0.0, 'std': 50, 'min': -250, 'max': 250},
        'Gy_w': {'mean': 0.0, 'std': 50, 'min': -250, 'max': 250},
        'Gz_w': {'mean': 0.0, 'std': 50, 'min': -250, 'max': 250},
    }


@pytest.fixture
def sample_monitoring_report():
    """Generate sample monitoring report for trigger policy tests."""
    return {
        'timestamp': '2026-01-30T12:00:00',
        'confidence_report': {
            'metrics': {
                'mean_confidence': 0.75,
                'std_confidence': 0.15,
                'mean_entropy': 1.2,
                'uncertain_ratio': 0.15
            }
        },
        'temporal_report': {
            'metrics': {
                'flip_rate': 0.18,
                'mean_dwell_time_seconds': 3.5,
                'short_dwell_ratio': 0.1
            }
        },
        'drift_report': {
            'per_channel_metrics': {
                'Ax_w': {'psi': 0.08, 'ks_statistic': 0.12},
                'Ay_w': {'psi': 0.05, 'ks_statistic': 0.08},
                'Az_w': {'psi': 0.06, 'ks_statistic': 0.10},
                'Gx_w': {'psi': 0.04, 'ks_statistic': 0.07},
                'Gy_w': {'psi': 0.03, 'ks_statistic': 0.05},
                'Gz_w': {'psi': 0.07, 'ks_statistic': 0.09}
            },
            'n_drifted_channels': 0,
            'aggregate_drift_score': 0.055,
            'overall_status': 'PASS'
        }
    }


@pytest.fixture
def degraded_monitoring_report():
    """Generate monitoring report showing model degradation."""
    return {
        'timestamp': '2026-01-30T12:00:00',
        'confidence_report': {
            'metrics': {
                'mean_confidence': 0.42,  # Below threshold
                'std_confidence': 0.25,
                'mean_entropy': 2.3,       # Above threshold
                'uncertain_ratio': 0.38    # Above threshold
            }
        },
        'temporal_report': {
            'metrics': {
                'flip_rate': 0.45,  # Above threshold
                'mean_dwell_time_seconds': 1.5,
                'short_dwell_ratio': 0.35
            }
        },
        'drift_report': {
            'per_channel_metrics': {
                'Ax_w': {'psi': 0.32, 'ks_statistic': 0.35},
                'Ay_w': {'psi': 0.28, 'ks_statistic': 0.30},
                'Az_w': {'psi': 0.35, 'ks_statistic': 0.38},
                'Gx_w': {'psi': 0.18, 'ks_statistic': 0.22},
                'Gy_w': {'psi': 0.15, 'ks_statistic': 0.18},
                'Gz_w': {'psi': 0.12, 'ks_statistic': 0.15}
            },
            'n_drifted_channels': 4,
            'aggregate_drift_score': 0.23,
            'overall_status': 'DRIFT'
        }
    }


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def mock_model_config():
    """Configuration for mock model."""
    return {
        'window_size': 200,
        'n_sensors': 6,
        'n_classes': 11,
        'input_shape': (200, 6)
    }


# ============================================================================
# HELPER FIXTURES
# ============================================================================

@pytest.fixture
def temp_state_file(tmp_path):
    """Create a temporary state file for trigger policy."""
    return tmp_path / "trigger_state.json"


@pytest.fixture
def activity_labels():
    """List of activity labels."""
    return [
        'ear_rubbing', 'forehead_rubbing', 'hair_pulling',
        'hand_scratching', 'hand_tapping', 'knuckles_cracking',
        'nail_biting', 'nape_rubbing', 'sitting', 'smoking', 'standing'
    ]
