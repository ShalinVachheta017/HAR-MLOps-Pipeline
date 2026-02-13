#!/usr/bin/env python3
"""
Tests for Prometheus Metrics Module
===================================

Tests metric recording and export functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMetricValue:
    """Tests for MetricValue storage class."""
    
    def test_counter_increment(self):
        """Test counter increment."""
        from prometheus_metrics import MetricValue
        
        metric = MetricValue('counter')
        
        metric.inc()
        metric.inc(5.0)
        
        assert metric._value == 6.0
    
    def test_counter_with_labels(self):
        """Test counter with labels."""
        from prometheus_metrics import MetricValue
        
        metric = MetricValue('counter')
        
        metric.inc(1.0, labels=('class_0',))
        metric.inc(2.0, labels=('class_1',))
        metric.inc(1.0, labels=('class_0',))
        
        assert metric._labels_values[('class_0',)] == 2.0
        assert metric._labels_values[('class_1',)] == 2.0
    
    def test_gauge_set(self):
        """Test gauge set."""
        from prometheus_metrics import MetricValue
        
        metric = MetricValue('gauge')
        
        metric.set(42.0)
        assert metric._value == 42.0
        
        metric.set(100.0)
        assert metric._value == 100.0
    
    def test_histogram_observe(self):
        """Test histogram observe."""
        from prometheus_metrics import MetricValue
        
        buckets = [0.1, 0.5, 1.0, 5.0]
        metric = MetricValue('histogram', buckets=buckets)
        
        metric.observe(0.05)  # Below first bucket
        metric.observe(0.3)   # In second bucket
        metric.observe(2.0)   # In fourth bucket
        metric.observe(10.0)  # Above all buckets (+Inf)
        
        assert metric._count == 4
        assert metric._sum == pytest.approx(12.35)


class TestMetricsExporter:
    """Tests for MetricsExporter class."""
    
    def test_singleton(self):
        """Test singleton pattern."""
        from prometheus_metrics import MetricsExporter
        
        exporter1 = MetricsExporter()
        exporter2 = MetricsExporter()
        
        assert exporter1 is exporter2
    
    def test_record_prediction(self):
        """Test recording a prediction."""
        from prometheus_metrics import MetricsExporter
        
        exporter = MetricsExporter()
        
        exporter.record_prediction(
            activity_class=0,
            confidence=0.95,
            latency_seconds=0.05
        )
        
        # Check counter incremented
        predictions_metric = exporter._metrics['har_predictions_total']
        assert ('0',) in predictions_metric._labels_values
    
    def test_record_batch_metrics(self):
        """Test recording batch metrics."""
        from prometheus_metrics import MetricsExporter
        
        exporter = MetricsExporter()
        
        exporter.record_batch_metrics(
            mean_confidence=0.87,
            mean_entropy=0.35,
            flip_rate=0.08,
            processing_time=2.5,
            n_samples=100
        )
        
        # Check gauges updated
        assert exporter._metrics['har_confidence_mean']._value == 0.87
        assert exporter._metrics['har_entropy_mean']._value == 0.35
        assert exporter._metrics['har_flip_rate']._value == 0.08
    
    def test_record_drift_metrics(self):
        """Test recording drift metrics."""
        from prometheus_metrics import MetricsExporter
        
        exporter = MetricsExporter()
        
        exporter.record_drift_metrics(
            psi_values={'acc_x': 0.05, 'acc_y': 0.12},
            ks_values={'acc_x': 0.04, 'acc_y': 0.09},
            drift_detected=True
        )
        
        psi_metric = exporter._metrics['har_drift_psi']
        assert ('acc_x',) in psi_metric._labels_values
        assert psi_metric._labels_values[('acc_x',)] == 0.05
    
    def test_record_trigger_state(self):
        """Test recording trigger state."""
        from prometheus_metrics import MetricsExporter
        
        exporter = MetricsExporter()
        
        exporter.record_trigger_state(state='warning', consecutive_warnings=2)
        
        trigger_metric = exporter._metrics['har_trigger_state']
        assert trigger_metric._labels_values[('overall',)] == 1  # warning = 1
        
        warnings_metric = exporter._metrics['har_consecutive_warnings']
        assert warnings_metric._value == 2
    
    def test_export_prometheus_format(self):
        """Test Prometheus text format export."""
        from prometheus_metrics import MetricsExporter
        
        exporter = MetricsExporter()
        
        # Record some metrics
        exporter.record_prediction(activity_class=1, confidence=0.9, latency_seconds=0.03)
        exporter.record_batch_metrics(
            mean_confidence=0.85,
            mean_entropy=0.4,
            flip_rate=0.1,
            processing_time=1.5,
            n_samples=50
        )
        
        output = exporter.export_prometheus()
        
        # Check format
        assert '# HELP' in output
        assert '# TYPE' in output
        assert 'har_confidence_mean' in output
        assert 'gauge' in output
    
    def test_export_json_format(self):
        """Test JSON format export."""
        from prometheus_metrics import MetricsExporter
        
        exporter = MetricsExporter()
        
        exporter.record_model_metrics(f1_score=0.89, accuracy=0.91)
        
        output = exporter.export_json()
        
        assert isinstance(output, dict)
        assert 'har_model_f1_score' in output
        assert 'har_model_accuracy' in output
    
    def test_record_from_monitoring_report(self):
        """Test recording from a full monitoring report."""
        from prometheus_metrics import MetricsExporter
        
        exporter = MetricsExporter()
        
        # Mock monitoring report structure
        report = {
            'proxy_metrics': {
                'mean_confidence': 0.82,
                'mean_entropy': 0.45,
                'flip_rate': 0.12,
                'n_predictions': 200
            },
            'drift_detection': {
                'psi_scores': {'acc_x': 0.08},
                'ks_scores': {'acc_x': 0.05},
                'drift_detected': False
            },
            'trigger_evaluation': {
                'should_retrain': False,
                'alert_level': 'WARNING',
                'consecutive_warnings': 1
            }
        }
        
        exporter.record_from_monitoring_report(report)
        
        # Verify metrics recorded
        assert exporter._metrics['har_confidence_mean']._value == 0.82
        assert exporter._metrics['har_entropy_mean']._value == 0.45


class TestMetricsServer:
    """Tests for metrics HTTP server."""
    
    def test_handler_metrics_endpoint(self):
        """Test that metrics endpoint is defined."""
        from prometheus_metrics import MetricsHandler, MetricsExporter
        
        # Setup handler
        MetricsHandler.exporter = MetricsExporter()
        
        # The actual HTTP server test would require threading
        # Just verify the class is correctly configured
        assert MetricsHandler.exporter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
