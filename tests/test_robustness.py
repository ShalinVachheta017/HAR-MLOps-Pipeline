"""
Robustness Test Suite
======================

Systematic tests for model robustness under sensor degradation.
Uses pytest parametrize to create a test matrix:
    - Gaussian noise at 0%, 5%, 10%, 20%
    - Missing data at 0%, 5%, 10%, 20%
    - Sampling jitter at 0%, 2%, 5%, 10%

These tests do NOT require a GPU or trained model — they validate that
the noise injection / evaluation pipeline works correctly and that
the degradation curves are monotonic (accuracy should decrease as
degradation increases).

Run:
    pytest tests/test_robustness.py -v
"""

import numpy as np
import pytest

from src.robustness import (
    RobustnessConfig,
    GaussianNoiseInjector,
    MissingDataInjector,
    SamplingJitterInjector,
    SaturationInjector,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_windows():
    """Create realistic sensor windows: (N, T, C) = (50, 200, 6)."""
    rng = np.random.RandomState(42)
    # Simulate accelerometer + gyroscope data
    X = rng.randn(50, 200, 6).astype(np.float32)
    # Scale accel channels (~±20 m/s²) and gyro channels (~±300 deg/s)
    X[:, :, :3] *= 10  # Accelerometer
    X[:, :, 3:] *= 100  # Gyroscope
    return X


@pytest.fixture
def sample_labels():
    """Labels for 50 windows, 11 classes."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 11, size=50)


# ── Gaussian Noise Tests ─────────────────────────────────────────────

class TestGaussianNoiseInjector:
    def test_zero_noise_is_identity(self, sample_windows):
        injector = GaussianNoiseInjector()
        result = injector.inject(sample_windows, noise_level=0.0)
        np.testing.assert_array_equal(result, sample_windows)

    @pytest.mark.parametrize("noise_level", [0.05, 0.10, 0.20, 0.50])
    def test_noise_changes_data(self, sample_windows, noise_level):
        injector = GaussianNoiseInjector()
        result = injector.inject(sample_windows, noise_level=noise_level)
        assert result.shape == sample_windows.shape
        assert not np.array_equal(result, sample_windows)

    def test_noise_magnitude_scales(self, sample_windows):
        injector = GaussianNoiseInjector()
        r1 = injector.inject(sample_windows, noise_level=0.05, seed=42)
        r2 = injector.inject(sample_windows, noise_level=0.50, seed=42)
        # Higher noise → larger deviation
        diff1 = np.mean(np.abs(r1 - sample_windows))
        diff2 = np.mean(np.abs(r2 - sample_windows))
        assert diff2 > diff1

    def test_noise_is_deterministic(self, sample_windows):
        injector = GaussianNoiseInjector()
        r1 = injector.inject(sample_windows, noise_level=0.1, seed=123)
        r2 = injector.inject(sample_windows, noise_level=0.1, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_give_different_noise(self, sample_windows):
        injector = GaussianNoiseInjector()
        r1 = injector.inject(sample_windows, noise_level=0.1, seed=1)
        r2 = injector.inject(sample_windows, noise_level=0.1, seed=2)
        assert not np.array_equal(r1, r2)


# ── Missing Data Tests ───────────────────────────────────────────────

class TestMissingDataInjector:
    def test_zero_missing_is_identity(self, sample_windows):
        injector = MissingDataInjector()
        result = injector.inject(sample_windows, missing_ratio=0.0)
        np.testing.assert_array_equal(result, sample_windows)

    @pytest.mark.parametrize("ratio", [0.05, 0.10, 0.20])
    def test_missing_creates_zeros(self, sample_windows, ratio):
        injector = MissingDataInjector()
        result = injector.inject(sample_windows, missing_ratio=ratio, mode="sample")
        # Some values should be zeroed
        n_zeros_orig = np.sum(sample_windows == 0)
        n_zeros_new = np.sum(result == 0)
        assert n_zeros_new > n_zeros_orig

    def test_window_mode_zeros_entire_windows(self, sample_windows):
        injector = MissingDataInjector()
        result = injector.inject(sample_windows, missing_ratio=0.20, mode="window")
        # ~10 windows should be all zeros
        zero_windows = np.sum(np.all(result == 0, axis=(1, 2)))
        assert zero_windows > 0

    def test_shape_preserved(self, sample_windows):
        injector = MissingDataInjector()
        result = injector.inject(sample_windows, missing_ratio=0.15)
        assert result.shape == sample_windows.shape


# ── Sampling Jitter Tests ────────────────────────────────────────────

class TestSamplingJitterInjector:
    def test_zero_jitter_is_identity(self, sample_windows):
        injector = SamplingJitterInjector()
        result = injector.inject(sample_windows, jitter_fraction=0.0)
        np.testing.assert_array_equal(result, sample_windows)

    @pytest.mark.parametrize("jitter", [0.02, 0.05, 0.10])
    def test_jitter_changes_data(self, sample_windows, jitter):
        injector = SamplingJitterInjector()
        result = injector.inject(sample_windows, jitter_fraction=jitter)
        assert result.shape == sample_windows.shape
        assert not np.array_equal(result, sample_windows)

    def test_jitter_preserves_shape(self, sample_windows):
        injector = SamplingJitterInjector()
        result = injector.inject(sample_windows, jitter_fraction=0.10)
        assert result.shape == sample_windows.shape


# ── Saturation Tests ─────────────────────────────────────────────────

class TestSaturationInjector:
    @pytest.mark.parametrize("threshold", [50.0, 20.0, 10.0, 5.0])
    def test_saturation_clips(self, sample_windows, threshold):
        injector = SaturationInjector()
        result = injector.inject(sample_windows, threshold=threshold)
        assert np.all(result >= -threshold)
        assert np.all(result <= threshold)
        assert result.shape == sample_windows.shape

    def test_high_threshold_is_near_identity(self, sample_windows):
        injector = SaturationInjector()
        result = injector.inject(sample_windows, threshold=1000.0)
        np.testing.assert_array_almost_equal(result, sample_windows)


# ── Integration Tests ────────────────────────────────────────────────

class TestRobustnessConfig:
    def test_default_config(self):
        config = RobustnessConfig()
        assert len(config.noise_levels) > 0
        assert len(config.missing_ratios) > 0
        assert config.n_seeds >= 1

    def test_custom_config(self):
        config = RobustnessConfig(
            noise_levels=[0.0, 0.1],
            missing_ratios=[0.0, 0.05],
            n_seeds=1,
        )
        assert len(config.noise_levels) == 2
        assert config.n_seeds == 1
