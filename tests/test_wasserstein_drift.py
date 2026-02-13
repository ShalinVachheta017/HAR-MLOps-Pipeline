"""
Wasserstein Drift Detection Tests
===================================

Tests for Wasserstein distance computation, drift detection,
and change-point detection.

Run:
    pytest tests/test_wasserstein_drift.py -v
"""

import numpy as np
import pytest

from src.wasserstein_drift import (
    WassersteinDriftConfig,
    WassersteinDriftDetector,
    WassersteinChangePointDetector,
)


@pytest.fixture
def baseline_data():
    """Baseline sensor data: (1000, 6)."""
    rng = np.random.RandomState(42)
    return rng.randn(1000, 6).astype(np.float32)


@pytest.fixture
def no_drift_data():
    """Production data from same distribution."""
    rng = np.random.RandomState(99)
    return rng.randn(1000, 6).astype(np.float32)


@pytest.fixture
def drifted_data():
    """Production data with clear drift (-mean shift + scale change)."""
    rng = np.random.RandomState(99)
    return (rng.randn(1000, 6) * 3 + 2).astype(np.float32)


# ── Wasserstein Distance ─────────────────────────────────────────────

class TestWassersteinDistance:
    def test_same_distribution_small_distance(self, baseline_data, no_drift_data):
        detector = WassersteinDriftDetector()
        dist = detector.wasserstein_1d(baseline_data[:, 0], no_drift_data[:, 0])
        assert dist < 0.2  # Same distribution → small distance

    def test_different_distribution_large_distance(self, baseline_data, drifted_data):
        detector = WassersteinDriftDetector()
        dist = detector.wasserstein_1d(baseline_data[:, 0], drifted_data[:, 0])
        assert dist > 1.0  # Shifted + scaled → large distance

    def test_identical_data_zero_distance(self, baseline_data):
        detector = WassersteinDriftDetector()
        dist = detector.wasserstein_1d(baseline_data[:, 0], baseline_data[:, 0])
        assert dist < 0.01


# ── Drift Detection ─────────────────────────────────────────────────

class TestWassersteinDriftDetector:
    def test_no_drift_detected(self, baseline_data, no_drift_data):
        detector = WassersteinDriftDetector()
        report = detector.detect(baseline_data, no_drift_data)
        assert report["overall_status"] == "NORMAL"
        assert report["n_channels_critical"] == 0

    def test_drift_detected(self, baseline_data, drifted_data):
        detector = WassersteinDriftDetector()
        report = detector.detect(baseline_data, drifted_data)
        assert report["overall_status"] in ["WARNING", "CRITICAL"]
        assert report["mean_wasserstein"] > 0.3

    def test_per_channel_keys(self, baseline_data, no_drift_data):
        detector = WassersteinDriftDetector()
        report = detector.detect(baseline_data, no_drift_data)
        assert "per_channel" in report
        assert len(report["per_channel"]) == 6  # 6 sensor channels

    def test_3d_input_works(self):
        """Test with windowed data (N, T, C)."""
        rng = np.random.RandomState(42)
        baseline = rng.randn(50, 200, 6).astype(np.float32)
        production = rng.randn(50, 200, 6).astype(np.float32)
        detector = WassersteinDriftDetector()
        report = detector.detect(baseline, production)
        assert "overall_status" in report

    def test_custom_thresholds(self, baseline_data, drifted_data):
        config = WassersteinDriftConfig(
            warn_threshold=0.1,
            critical_threshold=0.2,
        )
        detector = WassersteinDriftDetector(config)
        report = detector.detect(baseline_data, drifted_data)
        assert report["n_channels_critical"] > 0


# ── Change-Point Detection ───────────────────────────────────────────

class TestWassersteinChangePointDetector:
    def test_stable_series_no_change_points(self):
        """Constant Wasserstein distance → no change points."""
        series = np.ones(100) * 0.1 + np.random.RandomState(42).randn(100) * 0.01
        cpd = WassersteinChangePointDetector()
        result = cpd.detect_change_points(series)
        assert result["status"] == "STABLE"

    def test_step_change_detected(self):
        """Sudden jump in Wasserstein distance → change point."""
        series = np.concatenate([
            np.ones(60) * 0.1,
            np.ones(40) * 2.0,  # Sudden drift
        ])
        series += np.random.RandomState(42).randn(100) * 0.05
        cpd = WassersteinChangePointDetector()
        result = cpd.detect_change_points(series)
        assert result["status"] == "DRIFT_REGIME_CHANGE"
        assert len(result["change_points"]) > 0

    def test_insufficient_data(self):
        series = np.array([0.1, 0.2, 0.3])
        cpd = WassersteinChangePointDetector()
        result = cpd.detect_change_points(series)
        assert result["status"] == "INSUFFICIENT_DATA"
