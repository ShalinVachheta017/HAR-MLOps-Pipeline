"""
Calibration Module Tests
=========================

Tests for temperature scaling, ECE computation, and uncertainty analysis.

Run:
    pytest tests/test_calibration.py -v
"""

import numpy as np
import pytest

from src.calibration import (
    CalibrationConfig,
    TemperatureScaler,
    CalibrationEvaluator,
    UnlabeledCalibrationAnalyzer,
)


@pytest.fixture
def mock_logits():
    """Fake logits: (100, 11)."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 11).astype(np.float32)


@pytest.fixture
def mock_labels():
    """Ground-truth labels for 100 samples."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 11, size=100)


@pytest.fixture
def mock_probs(mock_logits):
    """Softmax probabilities from logits."""
    exp = np.exp(mock_logits - np.max(mock_logits, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


# ── Temperature Scaler ───────────────────────────────────────────────

class TestTemperatureScaler:
    def test_default_temperature(self):
        scaler = TemperatureScaler()
        assert scaler.temperature == 1.5
        assert not scaler.fitted

    def test_fit_returns_positive_temperature(self, mock_logits, mock_labels):
        scaler = TemperatureScaler()
        T = scaler.fit(mock_logits, mock_labels)
        assert T > 0
        assert scaler.fitted

    def test_transform_produces_valid_probs(self, mock_logits):
        scaler = TemperatureScaler()
        probs = scaler.transform(mock_logits)
        # Valid probability distribution
        assert probs.shape == mock_logits.shape
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_higher_temperature_softens_probs(self, mock_logits):
        scaler_low = TemperatureScaler(CalibrationConfig(initial_temperature=0.5))
        scaler_high = TemperatureScaler(CalibrationConfig(initial_temperature=5.0))

        probs_low = scaler_low.transform(mock_logits)
        probs_high = scaler_high.transform(mock_logits)

        # Higher temperature → more uniform → lower max confidence
        assert np.mean(np.max(probs_high, axis=1)) < np.mean(np.max(probs_low, axis=1))

    def test_save_load(self, mock_logits, mock_labels, tmp_path):
        scaler = TemperatureScaler()
        scaler.fit(mock_logits, mock_labels)
        path = tmp_path / "temperature.json"
        scaler.save(path)

        scaler2 = TemperatureScaler()
        scaler2.load(path)
        assert abs(scaler2.temperature - scaler.temperature) < 1e-6


# ── Calibration Evaluator ────────────────────────────────────────────

class TestCalibrationEvaluator:
    def test_ece_is_bounded(self, mock_probs, mock_labels):
        evaluator = CalibrationEvaluator(n_bins=15)
        ece, bins = evaluator.expected_calibration_error(mock_probs, mock_labels)
        assert 0 <= ece <= 1.0

    def test_brier_score_range(self, mock_probs, mock_labels):
        evaluator = CalibrationEvaluator()
        brier = evaluator.brier_score(mock_probs, mock_labels)
        assert 0 <= brier <= 2.0  # Multi-class Brier is bounded by 2

    def test_evaluate_returns_all_keys(self, mock_probs, mock_labels):
        evaluator = CalibrationEvaluator()
        report = evaluator.evaluate(mock_probs, mock_labels)
        assert "ece" in report
        assert "mce" in report
        assert "brier_score" in report
        assert "accuracy" in report
        assert "bins" in report
        assert len(report["bins"]) == 15

    def test_perfect_calibration_has_low_ece(self):
        """If predictions match labels perfectly, ECE should be low."""
        n = 1000
        rng = np.random.RandomState(42)
        labels = rng.randint(0, 5, size=n)
        probs = np.zeros((n, 5))
        probs[np.arange(n), labels] = 0.95
        probs += 0.05 / 5  # Small uniform noise
        probs /= probs.sum(axis=1, keepdims=True)

        evaluator = CalibrationEvaluator(n_bins=10)
        ece, _ = evaluator.expected_calibration_error(probs, labels)
        assert ece < 0.10

    def test_bins_count(self, mock_probs, mock_labels):
        evaluator = CalibrationEvaluator(n_bins=10)
        _, bins = evaluator.expected_calibration_error(mock_probs, mock_labels)
        assert len(bins) == 10


# ── Unlabeled Calibration Analyzer ───────────────────────────────────

class TestUnlabeledCalibrationAnalyzer:
    def test_analyze_returns_expected_keys(self, mock_probs):
        analyzer = UnlabeledCalibrationAnalyzer()
        result = analyzer.analyze(mock_probs)
        assert "mean_confidence" in result
        assert "mean_entropy" in result
        assert "overconfidence_ratio" in result
        assert "calibration_warnings" in result

    def test_overconfident_model_flagged(self):
        """A model with >80% confidence > 0.95 should trigger a warning."""
        n = 100
        probs = np.zeros((n, 5))
        probs[:, 0] = 0.98
        probs[:, 1:] = 0.005

        analyzer = UnlabeledCalibrationAnalyzer()
        result = analyzer.analyze(probs)
        assert result["overconfidence_ratio"] > 0.80
        assert len(result["calibration_warnings"]) > 0
