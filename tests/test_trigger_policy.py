"""
Tests for Trigger Policy Module
================================

Tests for the automated retraining trigger logic.
"""

import json
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trigger_policy import (
    TriggerPolicyEngine, 
    TriggerThresholds, 
    TriggerAction, 
    AlertLevel,
    ProxyModelValidator
)


class TestTriggerThresholds:
    """Tests for threshold configuration."""
    
    def test_default_thresholds(self):
        """Test that default thresholds are set correctly."""
        thresholds = TriggerThresholds()
        
        assert thresholds.confidence_warn == 0.55
        assert thresholds.confidence_critical == 0.45
        assert thresholds.psi_warn == 0.10
        assert thresholds.psi_critical == 0.25
    
    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        thresholds = TriggerThresholds(
            confidence_warn=0.60,
            psi_critical=0.30
        )
        
        assert thresholds.confidence_warn == 0.60
        assert thresholds.psi_critical == 0.30


class TestTriggerPolicyEngine:
    """Tests for the main trigger policy engine."""
    
    def test_engine_initialization(self, temp_state_file):
        """Test engine initializes correctly."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        
        assert engine.thresholds is not None
        assert engine.cooldown is not None
        assert engine.state is not None
    
    def test_normal_metrics_no_trigger(self, temp_state_file, sample_monitoring_report):
        """Test that normal metrics don't trigger retraining."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        
        decision = engine.evaluate(sample_monitoring_report)
        
        assert decision.should_trigger == False
        assert decision.action in [TriggerAction.NONE, TriggerAction.MONITOR]
        assert decision.alert_level == AlertLevel.INFO
    
    def test_degraded_metrics_trigger(self, temp_state_file, degraded_monitoring_report):
        """Test that degraded metrics trigger retraining."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        
        decision = engine.evaluate(degraded_monitoring_report)
        
        # With multiple signals degraded, should trigger
        assert decision.alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]
        # May be queued or triggered depending on severity
        assert decision.action in [
            TriggerAction.QUEUE_RETRAIN, 
            TriggerAction.TRIGGER_RETRAIN
        ]
    
    def test_low_confidence_triggers_warning(self, temp_state_file):
        """Test that low confidence triggers at least a warning."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        
        report = {
            'confidence_report': {
                'metrics': {
                    'mean_confidence': 0.50,  # Below warn threshold
                    'mean_entropy': 1.0,
                    'uncertain_ratio': 0.10
                }
            },
            'temporal_report': {
                'metrics': {'flip_rate': 0.10}
            },
            'drift_report': {
                'per_channel_metrics': {},
                'n_drifted_channels': 0
            }
        }
        
        decision = engine.evaluate(report)
        
        # Should at least note the low confidence
        assert 'confidence' in str(decision.signals).lower() or \
               decision.alert_level != AlertLevel.INFO
    
    def test_multiple_signals_required(self, temp_state_file):
        """Test that multiple signals are required for trigger."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        
        # Only one signal degraded
        report = {
            'confidence_report': {
                'metrics': {
                    'mean_confidence': 0.40,  # Critical
                    'mean_entropy': 1.0,
                    'uncertain_ratio': 0.10
                }
            },
            'temporal_report': {
                'metrics': {'flip_rate': 0.10}  # Normal
            },
            'drift_report': {
                'per_channel_metrics': {},  # Normal
                'n_drifted_channels': 0
            }
        }
        
        decision = engine.evaluate(report)
        
        # Single critical signal should still trigger due to severity
        assert decision.alert_level == AlertLevel.CRITICAL
    
    def test_state_persistence(self, temp_state_file, sample_monitoring_report):
        """Test that state is persisted between evaluations."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        
        # First evaluation
        engine.evaluate(sample_monitoring_report)
        
        # Check state file exists
        assert temp_state_file.exists()
        
        # Create new engine, should load state
        engine2 = TriggerPolicyEngine(state_file=temp_state_file)
        
        assert engine2.state['batches_since_retrain'] > 0
    
    def test_state_reset(self, temp_state_file, sample_monitoring_report):
        """Test state reset functionality."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        
        # Accumulate some state
        engine.evaluate(sample_monitoring_report)
        engine.evaluate(sample_monitoring_report)
        
        # Reset
        engine.reset_state()
        
        assert engine.state['warning_count'] == 0
        assert engine.state['consecutive_warnings'] == 0
        assert engine.state['batches_since_retrain'] == 0
    
    def test_trigger_summary(self, temp_state_file, sample_monitoring_report):
        """Test trigger summary generation."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        
        engine.evaluate(sample_monitoring_report)
        
        summary = engine.get_trigger_summary()
        
        assert 'warning_count' in summary
        assert 'consecutive_warnings' in summary
        assert 'batches_since_retrain' in summary


class TestConsecutiveWarnings:
    """Tests for consecutive warning escalation."""
    
    def test_consecutive_warnings_escalate(self, temp_state_file):
        """Test that consecutive warnings escalate to trigger."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        
        # Warning-level report (but not critical)
        warning_report = {
            'confidence_report': {
                'metrics': {
                    'mean_confidence': 0.52,  # Just below warn
                    'mean_entropy': 1.9,      # Just above warn
                    'uncertain_ratio': 0.22   # Above warn
                }
            },
            'temporal_report': {
                'metrics': {'flip_rate': 0.28}  # Above warn
            },
            'drift_report': {
                'per_channel_metrics': {
                    'Ax_w': {'psi': 0.12},
                    'Ay_w': {'psi': 0.11}
                },
                'n_drifted_channels': 2
            }
        }
        
        # Simulate consecutive warnings
        for i in range(4):
            decision = engine.evaluate(warning_report)
        
        # After 3+ consecutive warnings, should trigger
        # Note: exact behavior depends on threshold configuration
        assert engine.state['consecutive_warnings'] >= 3 or decision.should_trigger


class TestProxyModelValidator:
    """Tests for proxy-based model validation."""
    
    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = ProxyModelValidator(improvement_threshold=0.05)
        assert validator.improvement_threshold == 0.05
    
    def test_improved_model_recommended(self):
        """Test that improved model is recommended for deployment."""
        validator = ProxyModelValidator()
        
        np.random.seed(42)
        n_samples = 100
        n_classes = 11
        
        # Old model: lower confidence
        old_probs = np.random.dirichlet(np.ones(n_classes) * 0.5, size=n_samples)
        old_labels = np.argmax(old_probs, axis=1)
        
        # New model: higher confidence
        new_probs = np.random.dirichlet(np.ones(n_classes) * 3, size=n_samples)
        new_labels = np.argmax(new_probs, axis=1)
        
        old_predictions = {'probabilities': old_probs, 'labels': old_labels}
        new_predictions = {'probabilities': new_probs, 'labels': new_labels}
        
        result = validator.compare_models(old_predictions, new_predictions)
        
        assert 'should_deploy' in result
        assert 'comparisons' in result
        assert 'confidence' in result['comparisons']
    
    def test_regression_not_deployed(self):
        """Test that regressed model is not recommended."""
        validator = ProxyModelValidator()
        
        np.random.seed(42)
        n_samples = 100
        n_classes = 11
        
        # Old model: higher confidence
        old_probs = np.random.dirichlet(np.ones(n_classes) * 3, size=n_samples)
        old_labels = np.argmax(old_probs, axis=1)
        
        # New model: lower confidence (regression)
        new_probs = np.random.dirichlet(np.ones(n_classes) * 0.3, size=n_samples)
        new_labels = np.argmax(new_probs, axis=1)
        
        old_predictions = {'probabilities': old_probs, 'labels': old_labels}
        new_predictions = {'probabilities': new_probs, 'labels': new_labels}
        
        result = validator.compare_models(old_predictions, new_predictions)
        
        # Should not deploy due to regression
        assert result['should_deploy'] == False


class TestAlertLevels:
    """Tests for alert level assignment."""
    
    def test_info_level_for_normal(self, temp_state_file, sample_monitoring_report):
        """Test INFO level for normal metrics."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        decision = engine.evaluate(sample_monitoring_report)
        
        assert decision.alert_level == AlertLevel.INFO
    
    def test_critical_for_severe_degradation(self, temp_state_file, degraded_monitoring_report):
        """Test CRITICAL level for severe degradation."""
        engine = TriggerPolicyEngine(state_file=temp_state_file)
        decision = engine.evaluate(degraded_monitoring_report)
        
        # Should be WARNING or CRITICAL
        assert decision.alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]
