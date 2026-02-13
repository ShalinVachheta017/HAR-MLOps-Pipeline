#!/usr/bin/env python3
"""
Tests for OOD Detection Module
==============================

Tests energy-based and ensemble OOD detection.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestEnergyOODDetector:
    """Tests for energy-based OOD detection."""
    
    def test_compute_energy_from_logits(self):
        """Test energy computation from logits."""
        from ood_detection import EnergyOODDetector
        
        detector = EnergyOODDetector()
        
        # Create sample logits
        np.random.seed(42)
        logits = np.random.randn(100, 11)  # 11 HAR classes
        
        energy = detector.compute_energy(logits)
        
        assert energy.shape == (100,)
        assert np.all(np.isfinite(energy))
    
    def test_compute_energy_from_probs(self):
        """Test energy computation from probabilities."""
        from ood_detection import EnergyOODDetector
        
        detector = EnergyOODDetector()
        
        # Create softmax probabilities
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(11), size=100)
        
        energy = detector.compute_energy_from_probs(probs)
        
        assert energy.shape == (100,)
        assert np.all(energy >= 0)  # -log(max_prob) is non-negative
    
    def test_confident_predictions_have_lower_energy(self):
        """Test that confident predictions have lower energy."""
        from ood_detection import EnergyOODDetector
        
        detector = EnergyOODDetector()
        
        # Confident predictions (one class dominates)
        confident_probs = np.zeros((50, 11))
        confident_probs[:, 0] = 0.95
        confident_probs[:, 1:] = 0.05 / 10
        
        # Uncertain predictions (uniform-ish)
        uncertain_probs = np.ones((50, 11)) / 11
        
        confident_energy = detector.compute_energy_from_probs(confident_probs)
        uncertain_energy = detector.compute_energy_from_probs(uncertain_probs)
        
        # Confident should have lower energy (more in-distribution)
        assert np.mean(confident_energy) < np.mean(uncertain_energy)
    
    def test_calibration(self):
        """Test threshold calibration from validation data."""
        from ood_detection import EnergyOODDetector
        
        detector = EnergyOODDetector()
        
        # Simulate in-distribution energies
        np.random.seed(42)
        in_dist_energies = np.random.normal(-5.0, 1.0, size=1000)
        
        detector.calibrate(in_dist_energies)
        
        assert detector.in_dist_energy_mean is not None
        assert detector.in_dist_energy_std is not None
        assert detector.config.energy_threshold_warn > detector.in_dist_energy_mean
    
    def test_detect_ood(self):
        """Test OOD detection."""
        from ood_detection import EnergyOODDetector
        
        detector = EnergyOODDetector()
        
        # Mix of normal and OOD energies
        normal_energies = np.random.normal(-6.0, 0.5, size=80)
        ood_energies = np.random.normal(-2.0, 0.5, size=20)
        energies = np.concatenate([normal_energies, ood_energies])
        
        results = detector.detect_ood(energies)
        
        assert 'n_samples' in results
        assert 'n_normal' in results
        assert 'n_warning' in results
        assert 'n_critical' in results
        assert 'ood_ratio' in results
        assert results['n_samples'] == 100


class TestEnsembleOODDetector:
    """Tests for ensemble OOD detection."""
    
    def test_compute_all_scores(self):
        """Test computation of all OOD scores."""
        from ood_detection import EnsembleOODDetector
        
        detector = EnsembleOODDetector()
        
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(11), size=100)
        
        scores = detector.compute_all_scores(probs)
        
        assert 'energy' in scores
        assert 'entropy' in scores
        assert 'neg_confidence' in scores
        assert 'neg_margin' in scores
        
        for name, values in scores.items():
            assert values.shape == (100,)
    
    def test_ensemble_score(self):
        """Test ensemble score computation."""
        from ood_detection import EnsembleOODDetector
        
        detector = EnsembleOODDetector()
        
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(11), size=100)
        scores = detector.compute_all_scores(probs)
        
        ensemble = detector.compute_ensemble_score(scores)
        
        assert ensemble.shape == (100,)
        assert np.all(ensemble >= 0)
        assert np.all(ensemble <= 1)  # Normalized
    
    def test_full_detection(self):
        """Test full detection pipeline."""
        from ood_detection import EnsembleOODDetector
        
        detector = EnsembleOODDetector()
        
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(11), size=100)
        
        results = detector.detect(probs)
        
        assert 'n_samples' in results
        assert 'n_ood' in results
        assert 'ood_ratio' in results
        assert 'individual_scores' in results
        assert 'energy_breakdown' in results


class TestOODIntegration:
    """Integration tests for OOD detection."""
    
    def test_add_to_monitoring_report(self):
        """Test adding OOD metrics to monitoring report."""
        from ood_detection import add_ood_metrics_to_monitoring
        
        # Mock monitoring report
        report = {
            'timestamp': '2026-01-30T10:00:00',
            'proxy_metrics': {
                'mean_confidence': 0.85,
                'n_predictions': 100
            }
        }
        
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(11) * 5, size=100)
        
        updated_report = add_ood_metrics_to_monitoring(report, probs)
        
        assert 'ood_detection' in updated_report
        assert 'ensemble_ood_ratio' in updated_report['ood_detection']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
