"""
Tests for AdaBN (Adaptive Batch Normalization) domain adaptation.
"""

import pytest
import numpy as np


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def dummy_model():
    """Create a minimal Keras model with BatchNormalization layers."""
    tf = pytest.importorskip("tensorflow")
    keras = tf.keras

    inputs = keras.Input(shape=(200, 6))
    x = keras.layers.Conv1D(16, 3, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(11, activation="softmax")(x)
    model = keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


@pytest.fixture
def target_data():
    """Synthetic 'target domain' data: (100, 200, 6)."""
    np.random.seed(42)
    return np.random.randn(100, 200, 6).astype(np.float32)


# ── Tests ─────────────────────────────────────────────────────────────

class TestFindBNLayers:
    def test_finds_bn_layers(self, dummy_model):
        from src.domain_adaptation.adabn import _find_bn_layers
        layers = _find_bn_layers(dummy_model)
        assert len(layers) >= 1
        assert "batch_normalization" in layers[0].name.lower() or "bn" in layers[0].name.lower()

    def test_returns_empty_for_none(self):
        from src.domain_adaptation.adabn import _find_bn_layers
        assert _find_bn_layers(None) == []


class TestAdaptBNStatistics:
    def test_model_unchanged_weights(self, dummy_model, target_data):
        """AdaBN should NOT change convolutional / dense weights."""
        from src.domain_adaptation.adabn import adapt_bn_statistics
        tf = pytest.importorskip("tensorflow")

        # Capture kernel weights before
        conv_weights_before = dummy_model.layers[1].get_weights()[0].copy()

        adapt_bn_statistics(dummy_model, target_data, n_batches=3, batch_size=32)

        conv_weights_after = dummy_model.layers[1].get_weights()[0]
        np.testing.assert_array_equal(conv_weights_before, conv_weights_after)

    def test_bn_stats_change(self, dummy_model, target_data):
        """AdaBN should update BN running_mean / running_var."""
        from src.domain_adaptation.adabn import adapt_bn_statistics, _find_bn_layers

        bn_layers = _find_bn_layers(dummy_model)
        before_var = bn_layers[0].get_weights()[3].copy()  # running_variance starts at 1.0

        # Use reset_stats=True so we go from ones→something_else, and many batches
        adapt_bn_statistics(dummy_model, target_data, n_batches=20, batch_size=32, reset_stats=False)

        after_var = bn_layers[0].get_weights()[3]
        # Running variance should have shifted away from the initial value
        assert not np.allclose(before_var, after_var, atol=1e-4)

    def test_returns_model(self, dummy_model, target_data):
        from src.domain_adaptation.adabn import adapt_bn_statistics
        result = adapt_bn_statistics(dummy_model, target_data, n_batches=2)
        assert result is dummy_model


class TestAdaBNScoreConfidence:
    def test_returns_dict(self, dummy_model, target_data):
        from src.domain_adaptation.adabn import adabn_score_confidence
        result = adabn_score_confidence(dummy_model, target_data)
        assert isinstance(result, dict)
        assert "mean_confidence" in result
        assert "n_samples" in result
        assert result["n_samples"] == 100

    def test_confidence_range(self, dummy_model, target_data):
        from src.domain_adaptation.adabn import adabn_score_confidence
        result = adabn_score_confidence(dummy_model, target_data)
        assert 0.0 <= result["mean_confidence"] <= 1.0
        assert 0.0 <= result["low_confidence_ratio"] <= 1.0
