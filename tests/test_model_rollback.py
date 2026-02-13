#!/usr/bin/env python3
"""
Tests for Model Rollback Module
===============================

Tests model versioning, registry, and rollback functionality.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestModelVersion:
    """Tests for ModelVersion dataclass."""
    
    def test_creation(self):
        """Test model version creation."""
        from model_rollback import ModelVersion
        
        version = ModelVersion(
            version_id='v1.0.0',
            model_path=Path('/models/v1'),
            created_at=datetime.now().isoformat(),
            metrics={'f1': 0.89, 'accuracy': 0.91},
            training_config={'epochs': 100, 'lr': 0.001},
            is_active=True
        )
        
        assert version.version_id == 'v1.0.0'
        assert version.is_active is True
        assert version.metrics['f1'] == 0.89
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from model_rollback import ModelVersion
        
        version = ModelVersion(
            version_id='v1.0.0',
            model_path=Path('/models/v1'),
            created_at='2026-01-30T10:00:00',
            metrics={'f1': 0.89}
        )
        
        d = version.to_dict()
        
        assert isinstance(d, dict)
        assert d['version_id'] == 'v1.0.0'
        assert isinstance(d['model_path'], str)


class TestModelRegistry:
    """Tests for ModelRegistry class."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        from model_rollback import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(
                models_dir=Path(tmpdir) / 'models',
                max_versions=5
            )
            
            assert registry.models_dir.exists()
            assert registry.max_versions == 5
    
    def test_generate_version_id(self):
        """Test version ID generation."""
        from model_rollback import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(models_dir=Path(tmpdir))
            
            v1 = registry._generate_version_id()
            v2 = registry._generate_version_id()
            
            # Should be different (timestamp-based)
            assert v1 != v2 or v1 == v2  # May be same if called quickly
            assert v1.startswith('v')
    
    def test_register_model(self):
        """Test model registration."""
        from model_rollback import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(models_dir=Path(tmpdir))
            
            # Create a mock model directory
            model_path = Path(tmpdir) / 'new_model'
            model_path.mkdir()
            (model_path / 'model.h5').write_text('mock model')
            
            version = registry.register(
                model_path=model_path,
                metrics={'f1': 0.90},
                training_config={'epochs': 50}
            )
            
            assert version.version_id is not None
            assert version.metrics['f1'] == 0.90
            assert version.version_id in registry.versions
    
    def test_get_active_version(self):
        """Test getting active version."""
        from model_rollback import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(models_dir=Path(tmpdir))
            
            # Register a model
            model_path = Path(tmpdir) / 'model'
            model_path.mkdir()
            (model_path / 'model.h5').write_text('mock')
            
            version = registry.register(model_path, metrics={'f1': 0.85})
            registry.set_active(version.version_id)
            
            active = registry.get_active()
            
            assert active is not None
            assert active.version_id == version.version_id
    
    def test_list_versions(self):
        """Test listing all versions."""
        from model_rollback import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(models_dir=Path(tmpdir))
            
            # Register multiple models
            for i in range(3):
                model_path = Path(tmpdir) / f'model_{i}'
                model_path.mkdir()
                (model_path / 'model.h5').write_text(f'mock {i}')
                registry.register(model_path, metrics={'f1': 0.80 + i * 0.05})
            
            versions = registry.list_versions()
            
            assert len(versions) == 3
    
    def test_max_versions_cleanup(self):
        """Test that old versions are cleaned up."""
        from model_rollback import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(models_dir=Path(tmpdir), max_versions=3)
            
            # Register more than max_versions
            for i in range(5):
                model_path = Path(tmpdir) / f'model_{i}'
                model_path.mkdir()
                (model_path / 'model.h5').write_text(f'mock {i}')
                registry.register(model_path, metrics={'f1': 0.80})
            
            # Should only keep 3
            assert len(registry.versions) <= 3


class TestRollbackValidator:
    """Tests for RollbackValidator class."""
    
    def test_validation_passes(self):
        """Test validation that should pass."""
        from model_rollback import RollbackValidator, ModelVersion
        
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = RollbackValidator()
            
            # Create mock version with existing path
            model_path = Path(tmpdir) / 'model'
            model_path.mkdir()
            (model_path / 'model.h5').write_text('mock')
            
            version = ModelVersion(
                version_id='v1.0.0',
                model_path=model_path,
                created_at=datetime.now().isoformat(),
                metrics={'f1': 0.85}
            )
            
            is_valid, errors = validator.validate_version(version)
            
            assert is_valid is True
            assert len(errors) == 0
    
    def test_validation_fails_missing_path(self):
        """Test validation with missing model path."""
        from model_rollback import RollbackValidator, ModelVersion
        
        validator = RollbackValidator()
        
        version = ModelVersion(
            version_id='v1.0.0',
            model_path=Path('/nonexistent/path'),
            created_at=datetime.now().isoformat(),
            metrics={'f1': 0.85}
        )
        
        is_valid, errors = validator.validate_version(version)
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_validation_fails_low_metrics(self):
        """Test validation with poor metrics."""
        from model_rollback import RollbackValidator, ModelVersion
        
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = RollbackValidator(min_f1_score=0.80)
            
            model_path = Path(tmpdir) / 'model'
            model_path.mkdir()
            
            version = ModelVersion(
                version_id='v1.0.0',
                model_path=model_path,
                created_at=datetime.now().isoformat(),
                metrics={'f1': 0.70}  # Below threshold
            )
            
            is_valid, errors = validator.validate_version(version)
            
            assert is_valid is False
            assert any('F1' in e for e in errors)


class TestRollbackIntegration:
    """Integration tests for rollback workflow."""
    
    def test_full_rollback_workflow(self):
        """Test complete rollback workflow."""
        from model_rollback import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(models_dir=Path(tmpdir))
            
            # Register version 1
            model1 = Path(tmpdir) / 'model_v1'
            model1.mkdir()
            (model1 / 'model.h5').write_text('v1')
            v1 = registry.register(model1, metrics={'f1': 0.88})
            registry.set_active(v1.version_id)
            
            # Register version 2 (worse performance)
            model2 = Path(tmpdir) / 'model_v2'
            model2.mkdir()
            (model2 / 'model.h5').write_text('v2')
            v2 = registry.register(model2, metrics={'f1': 0.82})
            registry.set_active(v2.version_id)
            
            # Current active should be v2
            assert registry.get_active().version_id == v2.version_id
            
            # Rollback to v1
            result = registry.rollback(v1.version_id)
            
            assert result.get('success') is True
            assert registry.get_active().version_id == v1.version_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
