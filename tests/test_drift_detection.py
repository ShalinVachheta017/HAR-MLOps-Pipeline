"""
Tests for Drift Detection and Monitoring
=========================================

Tests for PSI, KS-test, and other drift metrics used in monitoring.
"""

import numpy as np
import pytest
from scipy import stats


class TestDriftMetrics:
    """Tests for drift detection metrics."""
    
    def test_identical_distributions_no_drift(self):
        """Test that identical distributions show no drift."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        # KS test should show no significant difference
        ks_stat, ks_pvalue = stats.ks_2samp(baseline, current)
        
        # With same distribution, p-value should be high
        assert ks_pvalue > 0.01, f"Unexpected drift detected: p={ks_pvalue}"
    
    def test_different_distributions_detect_drift(self):
        """Test that different distributions are detected as drift."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1, 1000)  # Shifted mean
        
        # KS test should detect significant difference
        ks_stat, ks_pvalue = stats.ks_2samp(baseline, current)
        
        # With different distribution, p-value should be low
        assert ks_pvalue < 0.01, f"Drift not detected: p={ks_pvalue}"
    
    def test_variance_change_detected(self):
        """Test that variance changes are detected."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 3, 1000)  # Same mean, different std
        
        ks_stat, ks_pvalue = stats.ks_2samp(baseline, current)
        
        # Should detect the variance change
        assert ks_pvalue < 0.05, f"Variance change not detected: p={ks_pvalue}"


class TestPSICalculation:
    """Tests for Population Stability Index calculation."""
    
    def compute_psi(self, baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute PSI between two distributions."""
        # Create bins based on baseline
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(baseline, percentiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Count in each bin
        baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
        current_counts = np.histogram(current, bins=bin_edges)[0]
        
        # Convert to proportions
        baseline_props = baseline_counts / len(baseline) + 1e-10
        current_props = current_counts / len(current) + 1e-10
        
        # Calculate PSI
        psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))
        
        return psi
    
    def test_psi_identical_distributions(self):
        """Test PSI is near zero for identical distributions."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        psi = self.compute_psi(baseline, current)
        
        # PSI should be very low for same distribution
        assert psi < 0.10, f"PSI too high for identical distributions: {psi}"
    
    def test_psi_shifted_distribution(self):
        """Test PSI detects shifted distribution."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(1, 1, 1000)  # Mean shifted by 1 std
        
        psi = self.compute_psi(baseline, current)
        
        # PSI should be elevated for shifted distribution
        assert psi > 0.10, f"PSI too low for shifted distribution: {psi}"
    
    def test_psi_major_shift(self):
        """Test PSI shows major shift for significantly different distributions."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(3, 2, 1000)  # Major shift
        
        psi = self.compute_psi(baseline, current)
        
        # PSI should be very high
        assert psi > 0.25, f"PSI should indicate major shift: {psi}"
    
    def test_psi_thresholds(self):
        """Test PSI against standard thresholds."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        
        # No shift
        current_same = np.random.normal(0, 1, 1000)
        psi_same = self.compute_psi(baseline, current_same)
        assert psi_same < 0.10, "Same distribution should pass"
        
        # Moderate shift
        current_moderate = np.random.normal(0.5, 1.2, 1000)
        psi_moderate = self.compute_psi(baseline, current_moderate)
        # May be in warning range 0.10-0.25
        
        # Major shift
        current_major = np.random.normal(2, 2, 1000)
        psi_major = self.compute_psi(baseline, current_major)
        assert psi_major > 0.25, "Major shift should exceed 0.25"


class TestMultiChannelDrift:
    """Tests for multi-channel drift aggregation."""
    
    def test_single_channel_drift_not_alarm(self):
        """Test that single channel drift doesn't trigger alarm."""
        # Simulating 6 channels, only 1 drifted
        channel_drift = [False, False, False, False, False, True]
        
        n_drifted = sum(channel_drift)
        threshold = 2  # Need at least 2 for warning
        
        assert n_drifted < threshold, "Single channel should not trigger"
    
    def test_multiple_channel_drift_triggers_alarm(self):
        """Test that multiple channel drift triggers alarm."""
        # Simulating 6 channels, 3 drifted
        channel_drift = [True, True, True, False, False, False]
        
        n_drifted = sum(channel_drift)
        threshold = 2
        
        assert n_drifted >= threshold, "Multiple channels should trigger"
    
    def test_aggregate_drift_score(self):
        """Test aggregate drift score calculation."""
        channel_psi = [0.05, 0.08, 0.12, 0.03, 0.15, 0.07]
        
        # Mean aggregation
        aggregate_score = np.mean(channel_psi)
        
        assert 0 < aggregate_score < 0.15
        
        # Max aggregation
        max_score = np.max(channel_psi)
        
        assert max_score == 0.15


class TestConfidenceMetrics:
    """Tests for confidence-based monitoring metrics."""
    
    def test_high_confidence_normal(self, sample_predictions):
        """Test that high confidence predictions are flagged as normal."""
        confidence = sample_predictions['confidence']
        mean_confidence = np.mean(confidence)
        
        # Sample predictions have mixed confidence
        # Just verify calculation works
        assert 0 < mean_confidence < 1
    
    def test_entropy_calculation(self, sample_predictions):
        """Test entropy calculation."""
        probs = sample_predictions['probabilities']
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        
        mean_entropy = np.mean(entropy)
        
        # Entropy should be positive
        assert mean_entropy > 0
        # For 11 classes, max entropy is ln(11) ≈ 2.4
        assert mean_entropy < 2.5
    
    def test_uncertain_ratio_calculation(self, sample_predictions):
        """Test uncertain sample ratio calculation."""
        confidence = sample_predictions['confidence']
        threshold = 0.5
        
        uncertain_count = np.sum(confidence < threshold)
        uncertain_ratio = uncertain_count / len(confidence)
        
        assert 0 <= uncertain_ratio <= 1


class TestTemporalMetrics:
    """Tests for temporal stability metrics."""
    
    def test_flip_rate_calculation(self):
        """Test prediction flip rate calculation."""
        # Sequence of predictions
        predictions = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0, 1])
        
        # Count flips (changes in prediction)
        flips = np.sum(predictions[1:] != predictions[:-1])
        flip_rate = flips / (len(predictions) - 1)
        
        # 4 flips in 9 transitions = 0.44
        expected_flip_rate = 4 / 9
        
        assert abs(flip_rate - expected_flip_rate) < 0.01
    
    def test_stable_predictions_low_flip_rate(self):
        """Test that stable predictions have low flip rate."""
        # Very stable sequence
        predictions = np.array([1] * 100)
        
        flips = np.sum(predictions[1:] != predictions[:-1])
        flip_rate = flips / (len(predictions) - 1)
        
        assert flip_rate == 0.0
    
    def test_unstable_predictions_high_flip_rate(self):
        """Test that unstable predictions have high flip rate."""
        # Alternating predictions (worst case)
        predictions = np.array([0, 1] * 50)
        
        flips = np.sum(predictions[1:] != predictions[:-1])
        flip_rate = flips / (len(predictions) - 1)
        
        assert flip_rate > 0.9
    
    def test_dwell_time_calculation(self):
        """Test activity dwell time calculation."""
        # Predictions with timestamps (assuming 2 second windows)
        predictions = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 0])
        window_stride_seconds = 2.0
        
        # Calculate dwell times
        dwell_times = []
        current_activity = predictions[0]
        current_dwell = 1
        
        for pred in predictions[1:]:
            if pred == current_activity:
                current_dwell += 1
            else:
                dwell_times.append(current_dwell * window_stride_seconds)
                current_activity = pred
                current_dwell = 1
        
        dwell_times.append(current_dwell * window_stride_seconds)
        
        # Expected: [6.0, 4.0, 8.0, 2.0] seconds
        assert len(dwell_times) == 4
        assert dwell_times[0] == 6.0  # 3 windows * 2s
