"""
Curriculum Pseudo-Labeling Tests
=================================

Tests for pseudo-label selection, threshold scheduling, and EWC.

Run:
    pytest tests/test_curriculum_pseudo_labeling.py -v
"""

import numpy as np
import pytest

from src.curriculum_pseudo_labeling import (
    CurriculumConfig,
    PseudoLabelSelector,
)


@pytest.fixture
def high_confidence_probs():
    """Probabilities where most predictions are very confident."""
    rng = np.random.RandomState(42)
    probs = np.zeros((200, 11))
    # Most samples have high confidence for their predicted class
    for i in range(200):
        cls = rng.randint(0, 11)
        probs[i, cls] = 0.90 + rng.random() * 0.09  # 0.90-0.99
        remaining = 1.0 - probs[i, cls]
        probs[i, :cls] = remaining / 10
        probs[i, cls+1:] = remaining / 10
    return probs


@pytest.fixture
def mixed_confidence_probs():
    """Mix of high and low confidence predictions."""
    rng = np.random.RandomState(42)
    probs = rng.dirichlet(np.ones(11), size=200)
    # Make ~50% high confidence
    for i in range(100):
        cls = rng.randint(0, 11)
        probs[i, cls] = 0.95
        remaining = 0.05
        probs[i, :cls] = remaining / 10
        probs[i, cls+1:] = remaining / 10
    return probs


# ── Threshold Scheduling ─────────────────────────────────────────────

class TestThresholdScheduling:
    def test_linear_decay(self):
        config = CurriculumConfig(
            initial_confidence_threshold=0.95,
            final_confidence_threshold=0.80,
            n_iterations=5,
            threshold_decay="linear",
        )
        selector = PseudoLabelSelector(config)

        thresholds = [selector.get_threshold_for_iteration(i) for i in range(5)]
        assert thresholds[0] == pytest.approx(0.95)
        assert thresholds[-1] == pytest.approx(0.80)
        # Should be monotonically decreasing
        for i in range(1, len(thresholds)):
            assert thresholds[i] <= thresholds[i - 1]

    def test_exponential_decay(self):
        config = CurriculumConfig(
            initial_confidence_threshold=0.95,
            final_confidence_threshold=0.80,
            n_iterations=5,
            threshold_decay="exponential",
        )
        selector = PseudoLabelSelector(config)

        thresholds = [selector.get_threshold_for_iteration(i) for i in range(5)]
        assert thresholds[0] == pytest.approx(0.95)
        assert thresholds[-1] == pytest.approx(0.80, abs=0.01)


# ── Pseudo-Label Selection ───────────────────────────────────────────

class TestPseudoLabelSelector:
    def test_select_at_high_threshold(self, high_confidence_probs):
        config = CurriculumConfig(initial_confidence_threshold=0.95)
        selector = PseudoLabelSelector(config)

        indices, labels, stats = selector.select(high_confidence_probs, iteration=0)
        assert stats["threshold"] == pytest.approx(0.95)
        # Should select some but not all
        assert len(indices) > 0
        assert len(indices) <= len(high_confidence_probs)

    def test_class_balancing(self, high_confidence_probs):
        config = CurriculumConfig(max_samples_per_class=5)
        selector = PseudoLabelSelector(config)

        indices, labels, stats = selector.select(high_confidence_probs, iteration=0)
        # No class should have more than 5 samples
        for cls, count in stats["per_class_selected"].items():
            assert count <= 5

    def test_more_selected_at_lower_threshold(self, mixed_confidence_probs):
        config = CurriculumConfig(
            initial_confidence_threshold=0.95,
            final_confidence_threshold=0.60,
            n_iterations=5,
            max_samples_per_class=100,  # Don't limit
        )
        selector = PseudoLabelSelector(config)

        _, _, stats_high = selector.select(mixed_confidence_probs, iteration=0)
        _, _, stats_low = selector.select(mixed_confidence_probs, iteration=4)

        assert stats_low["n_selected"] >= stats_high["n_selected"]

    def test_empty_selection_at_extreme_threshold(self):
        """If all predictions < threshold, nothing should be selected."""
        probs = np.full((50, 11), 1.0 / 11)  # Uniform → confidence ~0.09
        config = CurriculumConfig(initial_confidence_threshold=0.95)
        selector = PseudoLabelSelector(config)

        indices, labels, stats = selector.select(probs, iteration=0)
        assert stats["n_selected"] == 0

    def test_output_types(self, high_confidence_probs):
        selector = PseudoLabelSelector()
        indices, labels, stats = selector.select(high_confidence_probs, iteration=0)
        assert isinstance(indices, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(stats, dict)


# ── Config ───────────────────────────────────────────────────────────

class TestCurriculumConfig:
    def test_defaults(self):
        config = CurriculumConfig()
        assert config.initial_confidence_threshold == 0.95
        assert config.final_confidence_threshold == 0.80
        assert config.n_iterations == 5
        assert config.use_ewc is True

    def test_custom(self):
        config = CurriculumConfig(ewc_lambda=500.0, n_iterations=10)
        assert config.ewc_lambda == 500.0
        assert config.n_iterations == 10
