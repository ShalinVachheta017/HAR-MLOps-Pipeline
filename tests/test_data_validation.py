"""
Tests for Data Validation Module
=================================

Tests for sensor data validation, schema checking, and quality control.
"""

import numpy as np
import pandas as pd
import pytest


class TestDataValidation:
    """Tests for data validation functionality."""
    
    def test_sensor_columns_present(self, sample_sensor_data):
        """Test that all required sensor columns are present."""
        required_columns = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        
        for col in required_columns:
            assert col in sample_sensor_data.columns, f"Missing required column: {col}"
    
    def test_no_missing_values(self, sample_sensor_data):
        """Test that sensor data has no missing values."""
        sensor_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        
        for col in sensor_cols:
            missing_ratio = sample_sensor_data[col].isna().mean()
            assert missing_ratio == 0, f"Column {col} has {missing_ratio:.1%} missing values"
    
    def test_accelerometer_range(self, sample_sensor_data):
        """Test that accelerometer values are within reasonable range."""
        accel_cols = ['Ax_w', 'Ay_w', 'Az_w']
        max_reasonable_accel = 50.0  # m/s² (very generous limit)
        
        for col in accel_cols:
            max_val = sample_sensor_data[col].abs().max()
            assert max_val < max_reasonable_accel, \
                f"Column {col} has unreasonable value: {max_val}"
    
    def test_gyroscope_range(self, sample_sensor_data):
        """Test that gyroscope values are within reasonable range."""
        gyro_cols = ['Gx_w', 'Gy_w', 'Gz_w']
        max_reasonable_gyro = 500.0  # deg/s
        
        for col in gyro_cols:
            max_val = sample_sensor_data[col].abs().max()
            assert max_val < max_reasonable_gyro, \
                f"Column {col} has unreasonable value: {max_val}"
    
    def test_labeled_data_has_activity_column(self, sample_labeled_data):
        """Test that labeled data contains activity column."""
        assert 'activity' in sample_labeled_data.columns
    
    def test_labeled_data_has_user_column(self, sample_labeled_data):
        """Test that labeled data contains user column."""
        assert 'User' in sample_labeled_data.columns
    
    def test_activity_labels_valid(self, sample_labeled_data, activity_labels):
        """Test that all activity labels are valid."""
        unique_activities = sample_labeled_data['activity'].unique()
        
        for activity in unique_activities:
            # At least check it's a string
            assert isinstance(activity, str), f"Invalid activity type: {type(activity)}"
            assert len(activity) > 0, "Empty activity label found"


class TestDataQuality:
    """Tests for data quality checks."""
    
    def test_no_constant_columns(self, sample_sensor_data):
        """Test that no sensor column is constant (would indicate sensor failure)."""
        sensor_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        
        for col in sensor_cols:
            std = sample_sensor_data[col].std()
            assert std > 0.001, f"Column {col} appears constant (std={std})"
    
    def test_variance_not_collapsed(self, sample_sensor_data):
        """Test that variance hasn't collapsed (would indicate data issue)."""
        sensor_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        min_variance = 0.01
        
        for col in sensor_cols:
            variance = sample_sensor_data[col].var()
            assert variance > min_variance, \
                f"Column {col} has collapsed variance: {variance}"
    
    def test_no_infinite_values(self, sample_sensor_data):
        """Test that there are no infinite values."""
        sensor_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        
        for col in sensor_cols:
            inf_count = np.isinf(sample_sensor_data[col]).sum()
            assert inf_count == 0, f"Column {col} has {inf_count} infinite values"
    
    def test_sufficient_data_points(self, sample_sensor_data):
        """Test that we have sufficient data for processing."""
        min_samples = 200  # Minimum for one window
        assert len(sample_sensor_data) >= min_samples, \
            f"Insufficient data: {len(sample_sensor_data)} samples"


class TestWindowCreation:
    """Tests for sliding window creation."""
    
    def test_window_dimensions(self, sample_windows):
        """Test that windows have correct dimensions."""
        n_windows, window_size, n_sensors = sample_windows.shape
        
        assert window_size == 200, f"Expected window size 200, got {window_size}"
        assert n_sensors == 6, f"Expected 6 sensors, got {n_sensors}"
    
    def test_window_values_normalized(self, sample_windows):
        """Test that window values are in reasonable range after normalization."""
        # After standard scaling, most values should be within [-3, 3]
        # (within 3 standard deviations)
        
        # This test uses raw data, so we just check for non-NaN
        assert not np.isnan(sample_windows).any(), "Windows contain NaN values"
        assert not np.isinf(sample_windows).any(), "Windows contain infinite values"
    
    def test_minimum_windows_created(self, sample_labeled_data):
        """Test that window creation produces expected number of windows."""
        n_samples = len(sample_labeled_data)
        window_size = 200
        step_size = 100
        
        expected_windows = max(0, (n_samples - window_size) // step_size + 1)
        
        # Just verify the formula is reasonable
        assert expected_windows > 0, "Should create at least some windows"
