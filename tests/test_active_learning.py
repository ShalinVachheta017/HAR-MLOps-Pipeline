#!/usr/bin/env python3
"""
Tests for Active Learning Export Module
=======================================

Tests sample selection and export functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestUncertaintySampler:
    """Tests for uncertainty-based sampling."""
    
    def test_least_confidence(self):
        """Test least confidence scoring."""
        from active_learning_export import UncertaintySampler
        
        sampler = UncertaintySampler()
        
        # Confident predictions
        confident = np.array([[0.95, 0.03, 0.02]])
        # Uncertain predictions  
        uncertain = np.array([[0.4, 0.35, 0.25]])
        
        conf_score = sampler.least_confidence(confident)
        unc_score = sampler.least_confidence(uncertain)
        
        # Uncertain should have higher score
        assert unc_score[0] > conf_score[0]
    
    def test_entropy_sampling(self):
        """Test entropy-based scoring."""
        from active_learning_export import UncertaintySampler
        
        sampler = UncertaintySampler()
        
        # Low entropy (one class dominates)
        low_entropy = np.array([[0.99, 0.01/10] * 10 + [0.99]][:, :11])
        # High entropy (uniform)
        high_entropy = np.ones((1, 11)) / 11
        
        le_score = sampler.entropy_sampling(low_entropy)
        he_score = sampler.entropy_sampling(high_entropy)
        
        # High entropy should have higher score
        assert he_score[0] > le_score[0]
    
    def test_margin_sampling(self):
        """Test margin-based scoring."""
        from active_learning_export import UncertaintySampler
        
        sampler = UncertaintySampler()
        
        # Large margin
        large_margin = np.array([[0.9, 0.05, 0.05/9] * 9][:, :11])
        # Small margin (close decision)
        small_margin = np.array([[0.35, 0.33, 0.32/9] * 9][:, :11])
        
        lm_score = sampler.margin_sampling(large_margin)
        sm_score = sampler.margin_sampling(small_margin)
        
        # Small margin should have higher score
        assert sm_score[0] > lm_score[0]
    
    def test_select_samples(self):
        """Test sample selection."""
        from active_learning_export import UncertaintySampler
        
        sampler = UncertaintySampler()
        
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(11), size=500)
        predictions = np.argmax(probs, axis=1)
        
        indices, metadata = sampler.select_uncertain_samples(
            probs, n_samples=50, predictions=predictions
        )
        
        assert len(indices) <= 50
        assert metadata['n_selected'] == len(indices)
        assert 'mean_uncertainty' in metadata


class TestDiversitySampler:
    """Tests for diversity-based sampling."""
    
    def test_temporal_diversity(self):
        """Test temporal diversity sampling."""
        from active_learning_export import DiversitySampler
        
        sampler = DiversitySampler()
        
        # Create timestamps over 24 hours
        timestamps = np.arange(0, 86400, 100)  # 864 samples
        
        indices = sampler.temporal_diversity(timestamps, n_samples=50)
        
        assert len(indices) <= 50
        # Should be spread across time
        selected_times = timestamps[indices]
        assert np.std(selected_times) > 10000  # Good spread
    
    def test_prediction_diversity(self):
        """Test prediction diversity sampling."""
        from active_learning_export import DiversitySampler
        
        sampler = DiversitySampler()
        
        # Create imbalanced predictions
        predictions = np.array([0] * 100 + [1] * 50 + [2] * 20 + list(range(3, 11)) * 10)
        
        indices = sampler.prediction_diversity(predictions, n_samples=50)
        
        assert len(indices) <= 50
        # Should include samples from multiple classes
        selected_classes = set(predictions[indices])
        assert len(selected_classes) > 5


class TestActiveLearningExporter:
    """Tests for the main exporter class."""
    
    def test_select_samples_uncertainty(self):
        """Test uncertainty-based selection."""
        from active_learning_export import ActiveLearningExporter, ActiveLearningConfig
        
        config = ActiveLearningConfig(strategy='uncertainty')
        exporter = ActiveLearningExporter(config)
        
        np.random.seed(42)
        data = pd.DataFrame({
            'acc_x': np.random.randn(200),
            'acc_y': np.random.randn(200),
            'acc_z': np.random.randn(200)
        })
        probs = np.random.dirichlet(np.ones(11), size=200)
        preds = np.argmax(probs, axis=1)
        
        selected, metadata = exporter.select_samples(data, probs, preds, n_samples=30)
        
        assert len(selected) <= 30
        assert '_uncertainty_score' in selected.columns
        assert '_model_prediction' in selected.columns
    
    def test_select_samples_hybrid(self):
        """Test hybrid selection strategy."""
        from active_learning_export import ActiveLearningExporter, ActiveLearningConfig
        
        config = ActiveLearningConfig(strategy='hybrid')
        exporter = ActiveLearningExporter(config)
        
        np.random.seed(42)
        data = pd.DataFrame({'feature': np.random.randn(200)})
        probs = np.random.dirichlet(np.ones(11), size=200)
        preds = np.argmax(probs, axis=1)
        
        selected, metadata = exporter.select_samples(data, probs, preds, n_samples=50)
        
        assert metadata['strategy'] == 'hybrid'
        assert 'n_uncertain' in metadata
        assert 'n_diverse' in metadata
    
    def test_export_for_labeling(self):
        """Test export functionality."""
        from active_learning_export import ActiveLearningExporter, ActiveLearningConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ActiveLearningConfig(export_dir=Path(tmpdir))
            exporter = ActiveLearningExporter(config)
            
            np.random.seed(42)
            data = pd.DataFrame({
                'acc_x': np.random.randn(100),
                'acc_y': np.random.randn(100)
            })
            probs = np.random.dirichlet(np.ones(11), size=100)
            preds = np.argmax(probs, axis=1)
            
            selected, metadata = exporter.select_samples(data, probs, preds, n_samples=20)
            export_path = exporter.export_for_labeling(selected, metadata, batch_id='test_batch')
            
            assert export_path.exists()
            
            # Check metadata file
            metadata_path = export_path.parent / 'batch_metadata.json'
            assert metadata_path.exists()
            
            # Check instructions file
            instructions_path = export_path.parent / 'LABELING_INSTRUCTIONS.md'
            assert instructions_path.exists()
    
    def test_import_labeled_data(self):
        """Test import of labeled data."""
        from active_learning_export import ActiveLearningExporter, ActiveLearningConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ActiveLearningConfig(export_dir=Path(tmpdir))
            exporter = ActiveLearningExporter(config)
            
            # Create and export data
            np.random.seed(42)
            data = pd.DataFrame({'acc_x': np.random.randn(50)})
            probs = np.random.dirichlet(np.ones(11), size=50)
            preds = np.argmax(probs, axis=1)
            
            selected, metadata = exporter.select_samples(data, probs, preds, n_samples=20)
            export_path = exporter.export_for_labeling(selected, metadata, batch_id='import_test')
            
            # Simulate human labeling
            labeled_df = pd.read_csv(export_path)
            labeled_df['human_label'] = np.random.randint(0, 11, size=len(labeled_df))
            labeled_df.to_csv(export_path, index=False)
            
            # Import
            imported, stats = exporter.import_labeled_data(export_path.parent)
            
            assert stats['n_labeled'] == len(labeled_df)
            assert stats['completion_rate'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
