"""
Adaptive Batch Normalization (AdaBN) for unsupervised domain adaptation.

Idea:  The model's convolutional + LSTM layers learn domain-invariant features,
but the Batch-Normalisation layers overfit to the *source*-domain statistics
(mean/variance).  AdaBN simply replaces those running stats with the *target*
domain stats by doing a forward pass on unlabelled target data.

Reference:
    Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2018).
    "Revisiting Batch Normalization For Practical Domain Adaptation."
    arXiv:1603.04779

Usage:
    from src.domain_adaptation.adabn import adapt_bn_statistics

    adapted_model = adapt_bn_statistics(
        model,
        target_X,            # unlabelled production data, shape (N, T, C)
        n_batches=10,
        batch_size=64,
    )
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None


def _find_bn_layers(model):
    """Return all BatchNormalization layers in the model."""
    if model is None:
        return []
    bn_layers = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            bn_layers.append(layer)
    return bn_layers


def adapt_bn_statistics(
    model,
    target_X: np.ndarray,
    n_batches: int = 10,
    batch_size: int = 64,
    reset_stats: bool = True,
) -> "keras.Model":
    """
    Adapt Batch Normalization running statistics to a target domain.

    1.  Set all BN layers to *training* mode (so they update running stats).
    2.  Optionally reset running mean/var to zero/one.
    3.  Forward-pass `n_batches` of target data through the model.
    4.  Set all BN layers back to *inference* mode.

    The model weights (kernels, biases) are NOT updated — only the BN
    running_mean and running_variance.

    Parameters
    ----------
    model : keras.Model
        Pre-trained source model.
    target_X : np.ndarray, shape (N, window_size, n_channels)
        Unlabelled target-domain data.
    n_batches : int
        Number of mini-batches to use for statistics estimation.
    batch_size : int
        Mini-batch size.
    reset_stats : bool
        If True, zero-out running stats before adaptation.

    Returns
    -------
    model : keras.Model
        The same model object, with updated BN statistics.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for AdaBN.")

    bn_layers = _find_bn_layers(model)
    if not bn_layers:
        logger.warning("No BatchNormalization layers found — AdaBN is a no-op.")
        return model

    logger.info(
        "AdaBN: adapting %d BN layers with %d batches of size %d",
        len(bn_layers), n_batches, batch_size,
    )

    # ── 1. Reset running statistics ────────────────────────────────────
    if reset_stats:
        for layer in bn_layers:
            weights = layer.get_weights()
            # weights = [gamma, beta, running_mean, running_variance]
            gamma, beta = weights[0], weights[1]
            running_mean = np.zeros_like(weights[2])
            running_var = np.ones_like(weights[3])
            layer.set_weights([gamma, beta, running_mean, running_var])
        logger.debug("AdaBN: reset running stats for %d layers.", len(bn_layers))

    # ── 2. Store original training flags & set to training ─────────────
    original_training = {layer.name: layer.trainable for layer in model.layers}
    # Freeze all layers EXCEPT BN — BN must be trainable=True for
    # running_mean/var updates to take effect in TF2/Keras 3.
    for layer in model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    # ── 3. Forward pass on target batches ──────────────────────────────
    n_samples = target_X.shape[0]
    total_used = 0
    for i in range(n_batches):
        idx = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)
        batch = target_X[idx].astype(np.float32)
        # Forward pass with training=True so BN updates running stats
        _ = model(batch, training=True)
        total_used += len(idx)

    logger.info("AdaBN: completed adaptation using %d samples across %d batches.",
                total_used, n_batches)

    # ── 4. Restore original trainable flags ────────────────────────────
    for layer in model.layers:
        if layer.name in original_training:
            layer.trainable = original_training[layer.name]

    return model


def adabn_score_confidence(
    model,
    target_X: np.ndarray,
    batch_size: int = 64,
) -> dict:
    """
    Run inference on target data with the adapted model and return
    confidence statistics.  Useful for a quick proxy validation of
    the adaptation quality.
    """
    preds = model.predict(target_X, batch_size=batch_size, verbose=0)
    confidences = np.max(preds, axis=1)
    return {
        "mean_confidence": float(np.mean(confidences)),
        "median_confidence": float(np.median(confidences)),
        "min_confidence": float(np.min(confidences)),
        "std_confidence": float(np.std(confidences)),
        "low_confidence_ratio": float(np.mean(confidences < 0.5)),
        "n_samples": int(len(confidences)),
    }
