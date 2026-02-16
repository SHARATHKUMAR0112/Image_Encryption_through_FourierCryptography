"""
Unit tests for FourierTransformer.

Tests the core Fourier transform operations including DFT, IDFT,
sorting, and truncation of coefficients.
"""

import math

import numpy as np
import pytest

from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.models.exceptions import ImageProcessingError


class TestFourierTransformer:
    """Test suite for FourierTransformer class."""
    
    def test_compute_dft_simple_signal(self):
        """Test DFT on a simple known signal."""
        transformer = FourierTransformer()
        
        # Create a simple signal: a single frequency sine wave
        N = 8
        t = np.arange(N)
        # Signal: cos(2πt/N) + i*sin(2πt/N) = e^(i*2πt/N)
        points = np.exp(2j * np.pi * t / N)
        
        coefficients = transformer.compute_dft(points)
        
        # Verify we get N coefficients
        assert len(coefficients) == N
        
        # Verify all are FourierCoefficient objects
        assert all(isinstance(c, FourierCoefficient) for c in coefficients)
        
        # Verify frequencies are 0 to N-1
        frequencies = [c.frequency for c in coefficients]
        assert frequencies == list(range(N))
        
        # For this signal, frequency 1 should have the largest amplitude
        amplitudes = [c.amplitude for c in coefficients]
        max_amp_idx = amplitudes.index(max(amplitudes))
        assert coefficients[max_amp_idx].frequency == 1
    
    def test_compute_dft_empty_array(self):
        """Test that DFT raises error on empty array."""
        transformer = FourierTransformer()
        
        with pytest.raises(ImageProcessingError, match="cannot be empty"):
            transformer.compute_dft(np.array([]))
    
    def test_compute_dft_invalid_input(self):
        """Test that DFT raises error on invalid input."""
        transformer = FourierTransformer()
        
        with pytest.raises(ImageProcessingError, match="must be a NumPy array"):
            transformer.compute_dft([1, 2, 3])
    
    def test_compute_idft_reconstruction(self):
        """Test that IDFT reconstructs the original signal."""
        transformer = FourierTransformer()
        
        # Create random complex points
        N = 16
        original_points = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Compute DFT
        coefficients = transformer.compute_dft(original_points)
        
        # Compute IDFT
        reconstructed = transformer.compute_idft(coefficients)
        
        # Verify reconstruction matches original within numerical precision
        assert np.allclose(reconstructed, original_points, atol=1e-10)
    
    def test_compute_idft_empty_list(self):
        """Test that IDFT raises error on empty list."""
        transformer = FourierTransformer()
        
        with pytest.raises(ImageProcessingError, match="cannot be empty"):
            transformer.compute_idft([])
    
    def test_sort_by_amplitude(self):
        """Test sorting coefficients by amplitude."""
        transformer = FourierTransformer()
        
        # Create coefficients with known amplitudes
        coeffs = [
            FourierCoefficient(0, 5.0, 0.0, 5.0 + 0j),
            FourierCoefficient(1, 10.0, 0.0, 10.0 + 0j),
            FourierCoefficient(2, 3.0, 0.0, 3.0 + 0j),
            FourierCoefficient(3, 7.0, 0.0, 7.0 + 0j),
        ]
        
        sorted_coeffs = transformer.sort_by_amplitude(coeffs)
        
        # Verify sorted in descending order
        amplitudes = [c.amplitude for c in sorted_coeffs]
        assert amplitudes == [10.0, 7.0, 5.0, 3.0]
        
        # Verify frequencies are preserved
        frequencies = [c.frequency for c in sorted_coeffs]
        assert frequencies == [1, 3, 0, 2]
    
    def test_sort_by_amplitude_empty_list(self):
        """Test that sorting raises error on empty list."""
        transformer = FourierTransformer()
        
        with pytest.raises(ImageProcessingError, match="cannot be empty"):
            transformer.sort_by_amplitude([])
    
    def test_truncate_coefficients(self):
        """Test truncating to top N coefficients."""
        transformer = FourierTransformer()
        
        # Create 20 coefficients
        N = 20
        points = np.random.randn(N) + 1j * np.random.randn(N)
        coefficients = transformer.compute_dft(points)
        
        # Truncate to 10
        truncated = transformer.truncate_coefficients(coefficients, 10)
        
        # Verify we get exactly 10 coefficients
        assert len(truncated) == 10
        
        # Verify they are the top 10 by amplitude
        sorted_coeffs = transformer.sort_by_amplitude(coefficients)
        expected_amplitudes = [c.amplitude for c in sorted_coeffs[:10]]
        actual_amplitudes = [c.amplitude for c in truncated]
        assert sorted(actual_amplitudes, reverse=True) == sorted(expected_amplitudes, reverse=True)
    
    def test_truncate_coefficients_fewer_than_requested(self):
        """Test truncating when we have fewer coefficients than requested."""
        transformer = FourierTransformer()
        
        # Create 5 coefficients
        points = np.random.randn(5) + 1j * np.random.randn(5)
        coefficients = transformer.compute_dft(points)
        
        # Request 10 (more than we have)
        truncated = transformer.truncate_coefficients(coefficients, 10)
        
        # Should return all 5
        assert len(truncated) == 5
    
    def test_truncate_coefficients_invalid_range(self):
        """Test that truncate raises error for invalid num_terms."""
        transformer = FourierTransformer()
        
        points = np.random.randn(20) + 1j * np.random.randn(20)
        coefficients = transformer.compute_dft(points)
        
        # Test below minimum
        with pytest.raises(ImageProcessingError, match="must be in range"):
            transformer.truncate_coefficients(coefficients, 5)
        
        # Test above maximum
        with pytest.raises(ImageProcessingError, match="must be in range"):
            transformer.truncate_coefficients(coefficients, 1500)
    
    def test_dft_idft_round_trip_preserves_data(self):
        """Test that DFT followed by IDFT preserves original data."""
        transformer = FourierTransformer()
        
        # Create various test signals
        test_cases = [
            # Simple sine wave
            np.exp(2j * np.pi * np.arange(32) / 32),
            # Random complex signal
            np.random.randn(64) + 1j * np.random.randn(64),
            # Real signal (imaginary part is zero)
            np.random.randn(16) + 0j,
        ]
        
        for original in test_cases:
            coefficients = transformer.compute_dft(original)
            reconstructed = transformer.compute_idft(coefficients)
            
            assert np.allclose(reconstructed, original, atol=1e-10), \
                "Round-trip DFT/IDFT should preserve original signal"
    
    def test_dft_pure_sine_wave(self):
        """Test DFT on a pure sine wave with known frequency."""
        transformer = FourierTransformer()
        
        # Create a pure sine wave at frequency 3
        N = 64
        t = np.arange(N)
        frequency = 3
        # Signal: sin(2π*3*t/N) represented as complex exponential
        points = np.sin(2 * np.pi * frequency * t / N) + 0j
        
        coefficients = transformer.compute_dft(points)
        
        # Verify we get N coefficients
        assert len(coefficients) == N
        
        # For a pure sine wave, we expect peaks at frequency k and N-k
        # (positive and negative frequency components)
        amplitudes = [c.amplitude for c in coefficients]
        
        # Find the two largest amplitudes
        sorted_amps = sorted(enumerate(amplitudes), key=lambda x: x[1], reverse=True)
        top_two_indices = [sorted_amps[0][0], sorted_amps[1][0]]
        
        # Should be at frequency and N-frequency
        assert frequency in top_two_indices or (N - frequency) in top_two_indices
        
        # The sum of amplitudes at these two frequencies should dominate
        top_two_sum = sorted_amps[0][1] + sorted_amps[1][1]
        total_amplitude = sum(amplitudes)
        assert top_two_sum > 0.9 * total_amplitude
    
    def test_dft_cosine_wave(self):
        """Test DFT on a pure cosine wave with known frequency."""
        transformer = FourierTransformer()
        
        # Create a pure cosine wave at frequency 5
        N = 128
        t = np.arange(N)
        frequency = 5
        # Signal: cos(2π*5*t/N)
        points = np.cos(2 * np.pi * frequency * t / N) + 0j
        
        coefficients = transformer.compute_dft(points)
        
        # Verify we get N coefficients
        assert len(coefficients) == N
        
        # For a pure cosine wave, we expect peaks at frequency k and N-k
        amplitudes = [c.amplitude for c in coefficients]
        
        # Find peaks
        sorted_amps = sorted(enumerate(amplitudes), key=lambda x: x[1], reverse=True)
        top_two_indices = [sorted_amps[0][0], sorted_amps[1][0]]
        
        # Should be at frequency and N-frequency
        assert frequency in top_two_indices or (N - frequency) in top_two_indices
    
    def test_dft_dc_component(self):
        """Test DFT on a constant signal (DC component only)."""
        transformer = FourierTransformer()
        
        # Create a constant signal (DC component)
        N = 32
        constant_value = 5.0
        points = np.full(N, constant_value + 0j)
        
        coefficients = transformer.compute_dft(points)
        
        # Verify we get N coefficients
        assert len(coefficients) == N
        
        # For a DC signal, only frequency 0 should have significant amplitude
        # Frequency 0 amplitude should be N * constant_value
        assert coefficients[0].frequency == 0
        assert np.isclose(coefficients[0].amplitude, N * constant_value, atol=1e-10)
        
        # All other frequencies should have near-zero amplitude
        for i in range(1, N):
            assert coefficients[i].amplitude < 1e-10
    
    def test_truncate_minimum_coefficients(self):
        """Test truncating to minimum allowed coefficient count (10)."""
        transformer = FourierTransformer()
        
        # Create signal with many coefficients
        N = 100
        points = np.random.randn(N) + 1j * np.random.randn(N)
        coefficients = transformer.compute_dft(points)
        
        # Truncate to minimum (10)
        truncated = transformer.truncate_coefficients(coefficients, 10)
        
        # Verify we get exactly 10 coefficients
        assert len(truncated) == 10
        
        # Verify they are the top 10 by amplitude
        sorted_coeffs = transformer.sort_by_amplitude(coefficients)
        expected_top_10 = sorted_coeffs[:10]
        
        # Check that truncated contains the same coefficients (by amplitude)
        truncated_amps = sorted([c.amplitude for c in truncated], reverse=True)
        expected_amps = sorted([c.amplitude for c in expected_top_10], reverse=True)
        assert np.allclose(truncated_amps, expected_amps, atol=1e-10)
    
    def test_truncate_exactly_minimum_coefficients(self):
        """Test behavior when signal has exactly 10 coefficients."""
        transformer = FourierTransformer()
        
        # Create signal with exactly 10 points
        N = 10
        points = np.random.randn(N) + 1j * np.random.randn(N)
        coefficients = transformer.compute_dft(points)
        
        # Truncate to 10 (same as input)
        truncated = transformer.truncate_coefficients(coefficients, 10)
        
        # Should return all 10 coefficients
        assert len(truncated) == 10
        
        # All original coefficients should be present
        original_amps = sorted([c.amplitude for c in coefficients], reverse=True)
        truncated_amps = sorted([c.amplitude for c in truncated], reverse=True)
        assert np.allclose(truncated_amps, original_amps, atol=1e-10)
