"""
Tests for Preprocessing Module
===============================

Tests for data preprocessing including gravity removal, normalization, etc.
"""

import numpy as np
import pytest
from scipy import signal


class TestGravityRemoval:
    """Tests for gravity removal from accelerometer data."""
    
    def apply_highpass_filter(self, data: np.ndarray, cutoff: float = 0.3, fs: float = 50.0) -> np.ndarray:
        """Apply high-pass Butterworth filter to remove gravity."""
        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
        
        b, a = signal.butter(3, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, data)
    
    def test_gravity_removal_reduces_dc_offset(self):
        """Test that gravity removal reduces DC offset in z-axis."""
        np.random.seed(42)
        
        # Simulate accelerometer data with gravity (9.8 m/s² on z-axis)
        n_samples = 1000
        fs = 50.0  # 50 Hz
        
        # Z-axis with gravity + motion
        az_with_gravity = 9.8 + np.random.normal(0, 0.5, n_samples)
        
        # Apply high-pass filter
        az_filtered = self.apply_highpass_filter(az_with_gravity, cutoff=0.3, fs=fs)
        
        # After filtering, mean should be close to zero
        assert abs(np.mean(az_filtered)) < 1.0, \
            f"Gravity not removed: mean={np.mean(az_filtered)}"
    
    def test_gravity_removal_preserves_motion(self):
        """Test that gravity removal preserves dynamic motion signals."""
        np.random.seed(42)
        
        n_samples = 1000
        fs = 50.0
        t = np.arange(n_samples) / fs
        
        # Simulated motion: 2 Hz oscillation (dynamic activity)
        motion = 2.0 * np.sin(2 * np.pi * 2 * t)
        
        # Add gravity
        az_with_gravity = 9.8 + motion
        
        # Apply filter
        az_filtered = self.apply_highpass_filter(az_with_gravity, cutoff=0.3, fs=fs)
        
        # Motion signal should be preserved (check correlation)
        # Skip edges due to filter transients
        correlation = np.corrcoef(motion[100:-100], az_filtered[100:-100])[0, 1]
        
        assert correlation > 0.9, f"Motion signal not preserved: corr={correlation}"
    
    def test_static_data_zeroed(self):
        """Test that purely static data (just gravity) is zeroed."""
        np.random.seed(42)
        
        n_samples = 1000
        fs = 50.0
        
        # Pure gravity signal (no motion)
        az_static = np.full(n_samples, 9.8)
        
        # Apply filter
        az_filtered = self.apply_highpass_filter(az_static, cutoff=0.3, fs=fs)
        
        # Should be essentially zero (except for filter edge effects)
        # Check middle portion
        mid_section = az_filtered[200:-200]
        
        assert np.abs(mid_section).max() < 0.5, \
            f"Static signal not zeroed: max={np.abs(mid_section).max()}"


class TestNormalization:
    """Tests for data normalization."""
    
    def test_standard_scaling(self, sample_sensor_data):
        """Test standard scaling (zero mean, unit variance)."""
        from sklearn.preprocessing import StandardScaler
        
        sensor_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        data = sample_sensor_data[sensor_cols].values
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Check mean ≈ 0
        for i, col in enumerate(sensor_cols):
            mean = np.mean(scaled_data[:, i])
            assert abs(mean) < 0.01, f"Mean not zero for {col}: {mean}"
        
        # Check std ≈ 1
        for i, col in enumerate(sensor_cols):
            std = np.std(scaled_data[:, i])
            assert abs(std - 1.0) < 0.01, f"Std not one for {col}: {std}"
    
    def test_normalization_preserves_shape(self, sample_sensor_data):
        """Test that normalization preserves data shape."""
        from sklearn.preprocessing import StandardScaler
        
        sensor_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        data = sample_sensor_data[sensor_cols].values
        
        original_shape = data.shape
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        assert scaled_data.shape == original_shape
    
    def test_scaler_parameters_stored(self):
        """Test that scaler parameters can be stored and restored."""
        from sklearn.preprocessing import StandardScaler
        import json
        
        np.random.seed(42)
        data = np.random.randn(1000, 6)
        
        scaler = StandardScaler()
        scaler.fit(data)
        
        # Store parameters
        params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
        
        # Simulate saving and loading
        params_json = json.dumps(params)
        loaded_params = json.loads(params_json)
        
        # Restore scaler
        new_scaler = StandardScaler()
        new_scaler.mean_ = np.array(loaded_params['mean'])
        new_scaler.scale_ = np.array(loaded_params['scale'])
        new_scaler.var_ = new_scaler.scale_ ** 2
        new_scaler.n_features_in_ = len(loaded_params['mean'])
        
        # Check parameters match
        np.testing.assert_array_almost_equal(scaler.mean_, new_scaler.mean_)
        np.testing.assert_array_almost_equal(scaler.scale_, new_scaler.scale_)


class TestResampling:
    """Tests for data resampling."""
    
    def test_upsample_maintains_length(self):
        """Test that upsampling increases data length appropriately."""
        np.random.seed(42)
        
        original_samples = 100
        original_hz = 25
        target_hz = 50
        
        data = np.random.randn(original_samples, 6)
        
        # Calculate expected length
        duration = original_samples / original_hz
        expected_samples = int(duration * target_hz)
        
        # Simple linear interpolation
        x_original = np.linspace(0, duration, original_samples)
        x_resampled = np.linspace(0, duration, expected_samples)
        
        resampled = np.zeros((expected_samples, 6))
        for i in range(6):
            resampled[:, i] = np.interp(x_resampled, x_original, data[:, i])
        
        assert resampled.shape[0] == expected_samples
    
    def test_downsample_maintains_length(self):
        """Test that downsampling decreases data length appropriately."""
        np.random.seed(42)
        
        original_samples = 200
        original_hz = 100
        target_hz = 50
        
        data = np.random.randn(original_samples, 6)
        
        # Calculate expected length
        duration = original_samples / original_hz
        expected_samples = int(duration * target_hz)
        
        # Simple decimation
        from scipy import signal as sig
        
        # Resample factor
        factor = target_hz / original_hz
        
        resampled = sig.resample(data, expected_samples, axis=0)
        
        assert resampled.shape[0] == expected_samples


class TestUnitConversion:
    """Tests for unit conversion (e.g., milliG to m/s²)."""
    
    def test_millig_to_ms2_conversion(self):
        """Test conversion from milliG to m/s²."""
        # 1 G = 9.80665 m/s²
        # 1 milliG = 0.00980665 m/s²
        
        millig_value = 1000  # 1 G
        expected_ms2 = 9.80665
        
        conversion_factor = 9.80665 / 1000
        converted = millig_value * conversion_factor
        
        assert abs(converted - expected_ms2) < 0.001
    
    def test_conversion_preserves_sign(self):
        """Test that unit conversion preserves sign."""
        millig_values = np.array([-1000, 0, 1000, 2000])
        
        conversion_factor = 9.80665 / 1000
        converted = millig_values * conversion_factor
        
        assert converted[0] < 0  # Negative preserved
        assert converted[1] == 0  # Zero preserved
        assert converted[2] > 0  # Positive preserved
        assert converted[3] > converted[2]  # Ordering preserved


class TestDataIntegrity:
    """Tests for data integrity during preprocessing."""
    
    def test_no_nan_introduced(self, sample_sensor_data):
        """Test that preprocessing doesn't introduce NaN values."""
        from sklearn.preprocessing import StandardScaler
        
        sensor_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        data = sample_sensor_data[sensor_cols].values
        
        # Verify no NaN in input
        assert not np.isnan(data).any(), "Input data contains NaN"
        
        # Normalize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Check no NaN introduced
        assert not np.isnan(scaled_data).any(), "Normalization introduced NaN"
    
    def test_no_inf_introduced(self, sample_sensor_data):
        """Test that preprocessing doesn't introduce infinite values."""
        from sklearn.preprocessing import StandardScaler
        
        sensor_cols = ['Ax_w', 'Ay_w', 'Az_w', 'Gx_w', 'Gy_w', 'Gz_w']
        data = sample_sensor_data[sensor_cols].values
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        assert not np.isinf(scaled_data).any(), "Normalization introduced Inf"
