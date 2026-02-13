"""
Sensor Placement Module Tests
==============================

Tests for axis mirroring augmentation, hand detection, and
per-hand performance reporting.

Run:
    pytest tests/test_sensor_placement.py -v
"""

import numpy as np
import pytest

from src.sensor_placement import (
    SensorPlacementConfig,
    AxisMirrorAugmenter,
    HandDetector,
    HandPerformanceReporter,
)


@pytest.fixture
def sample_windows():
    """Sensor windows: (50, 200, 6)."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 200, 6).astype(np.float32)
    X[:, :, :3] *= 10  # Accel
    X[:, :, 3:] *= 100  # Gyro
    return X


@pytest.fixture
def sample_labels():
    rng = np.random.RandomState(42)
    return rng.randint(0, 11, size=50)


# ── Axis Mirror Augmenter ────────────────────────────────────────────

class TestAxisMirrorAugmenter:
    def test_mirror_flips_correct_axes(self, sample_windows):
        augmenter = AxisMirrorAugmenter()
        mirrored = augmenter.mirror(sample_windows)
        # Axes 1,2,4,5 should be negated
        for ax in [1, 2, 4, 5]:
            np.testing.assert_array_almost_equal(
                mirrored[:, :, ax], -sample_windows[:, :, ax]
            )
        # Axes 0,3 should be unchanged
        for ax in [0, 3]:
            np.testing.assert_array_almost_equal(
                mirrored[:, :, ax], sample_windows[:, :, ax]
            )

    def test_augment_increases_dataset(self, sample_windows, sample_labels):
        augmenter = AxisMirrorAugmenter()
        X_aug, y_aug = augmenter.augment(sample_windows, sample_labels, probability=1.0)
        assert len(X_aug) == 2 * len(sample_windows)
        assert len(y_aug) == 2 * len(sample_labels)

    def test_augment_zero_probability(self, sample_windows, sample_labels):
        augmenter = AxisMirrorAugmenter()
        X_aug, y_aug = augmenter.augment(sample_windows, sample_labels, probability=0.0)
        assert len(X_aug) == len(sample_windows)

    def test_augment_preserves_shape(self, sample_windows, sample_labels):
        augmenter = AxisMirrorAugmenter()
        X_aug, y_aug = augmenter.augment(sample_windows, sample_labels)
        assert X_aug.shape[1:] == sample_windows.shape[1:]

    def test_custom_mirror_axes(self, sample_windows, sample_labels):
        config = SensorPlacementConfig(mirror_axes=[0, 1])
        augmenter = AxisMirrorAugmenter(config)
        mirrored = augmenter.mirror(sample_windows)
        np.testing.assert_array_almost_equal(
            mirrored[:, :, 0], -sample_windows[:, :, 0]
        )


# ── Hand Detector ────────────────────────────────────────────────────

class TestHandDetector:
    def test_detect_returns_keys(self, sample_windows):
        detector = HandDetector()
        result = detector.detect(sample_windows)
        assert "detected_hand" in result
        assert "detection_confidence" in result
        assert "features" in result

    def test_detect_with_reference(self, sample_windows):
        detector = HandDetector()
        ref_stats = {"total_accel_variance": 50.0}
        result = detector.detect(sample_windows, reference_stats=ref_stats)
        assert result["detected_hand"] in ["DOMINANT", "NON_DOMINANT", "AMBIGUOUS"]

    def test_detect_3d_and_2d(self, sample_windows):
        detector = HandDetector()
        # 3D input
        r1 = detector.detect(sample_windows)
        # 2D input
        r2 = detector.detect(sample_windows.reshape(-1, 6))
        assert "detected_hand" in r1
        assert "detected_hand" in r2


# ── Hand Performance Reporter ────────────────────────────────────────

class TestHandPerformanceReporter:
    def test_report_with_labels(self):
        reporter = HandPerformanceReporter()
        predictions = np.array([0, 1, 2, 0, 1, 2])
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        hand_labels = np.array(["DOMINANT", "DOMINANT", "DOMINANT",
                                "NON_DOMINANT", "NON_DOMINANT", "NON_DOMINANT"])
        true_labels = np.array([0, 1, 2, 0, 0, 2])

        result = reporter.report(predictions, confidences, hand_labels, true_labels)
        assert "DOMINANT" in result
        assert "NON_DOMINANT" in result
        assert "accuracy" in result["DOMINANT"]

    def test_report_without_labels(self):
        reporter = HandPerformanceReporter()
        predictions = np.array([0, 1, 2, 0])
        confidences = np.array([0.9, 0.8, 0.7, 0.6])
        hand_labels = np.array(["DOMINANT", "DOMINANT", "NON_DOMINANT", "NON_DOMINANT"])

        result = reporter.report(predictions, confidences, hand_labels)
        assert "DOMINANT" in result
        assert "accuracy" not in result["DOMINANT"]
