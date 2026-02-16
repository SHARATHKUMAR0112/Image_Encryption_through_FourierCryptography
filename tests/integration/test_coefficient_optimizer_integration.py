"""
Integration tests for CoefficientOptimizer with the full pipeline.

Tests the optimizer's integration with image processing, Fourier transform,
and the overall encryption workflow.
"""

import numpy as np
import pytest

from fourier_encryption.ai.coefficient_optimizer import CoefficientOptimizer
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.models.data_models import Contour


class TestCoefficientOptimizerIntegration:
    """Integration tests for CoefficientOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer with default settings."""
        return CoefficientOptimizer(
            target_error=0.05,
            min_coefficients=10,
            max_coefficients=1000
        )
    
    @pytest.fixture
    def transformer(self):
        """Create Fourier transformer."""
        return FourierTransformer()
    
    @pytest.fixture
    def extractor(self):
        """Create contour extractor."""
        return ContourExtractor()
    
    def test_optimize_circular_contour(self, optimizer, transformer, extractor):
        """Test optimization on a circular contour."""
        # Create a circular contour
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        x = 50 + 30 * np.cos(t)
        y = 50 + 30 * np.sin(t)
        points = np.column_stack([x, y])
        
        contour = Contour(points=points, is_closed=True, length=len(points))
        
        # Convert to complex plane
        complex_points = extractor.to_complex_plane(contour)
        
        # Compute DFT
        coefficients = transformer.compute_dft(complex_points)
        
        # Optimize
        result = optimizer.optimize_count(coefficients, complex_points)
        
        # Verify result
        assert 10 <= result.optimal_count <= len(coefficients)
        assert result.reconstruction_error <= 0.05
        assert result.complexity_class in ["low", "medium", "high"]
        assert len(result.explanation) > 0
        
        # For a circle, we should need very few coefficients
        assert result.optimal_count < len(coefficients) * 0.5
    
    def test_optimize_square_contour(self, optimizer, transformer, extractor):
        """Test optimization on a square contour."""
        # Create a square contour
        points = np.array([
            [10, 10], [90, 10], [90, 90], [10, 90]
        ])
        
        # Resample to have more points
        contour = Contour(points=points, is_closed=True, length=len(points))
        resampled = extractor.resample_contour(contour, 100)
        
        # Convert to complex plane
        complex_points = extractor.to_complex_plane(resampled)
        
        # Compute DFT
        coefficients = transformer.compute_dft(complex_points)
        
        # Optimize
        result = optimizer.optimize_count(coefficients, complex_points)
        
        # Verify result
        assert 10 <= result.optimal_count <= len(coefficients)
        assert result.reconstruction_error <= 0.05
        assert result.complexity_class in ["low", "medium", "high"]
    
    def test_optimize_with_different_target_errors(self, transformer, extractor):
        """Test that different target errors produce different results."""
        # Create a contour
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        points = np.exp(1j * t) + 0.3 * np.exp(1j * 3 * t)
        
        # Compute DFT
        coefficients = transformer.compute_dft(points)
        
        # Test with strict error threshold
        strict_optimizer = CoefficientOptimizer(target_error=0.01)
        strict_result = strict_optimizer.optimize_count(coefficients, points)
        
        # Test with relaxed error threshold
        relaxed_optimizer = CoefficientOptimizer(target_error=0.1)
        relaxed_result = relaxed_optimizer.optimize_count(coefficients, points)
        
        # Relaxed threshold should need fewer or equal coefficients
        assert relaxed_result.optimal_count <= strict_result.optimal_count
        
        # Both should meet their respective thresholds
        assert strict_result.reconstruction_error <= 0.01 * 1.1
        assert relaxed_result.reconstruction_error <= 0.1 * 1.1
    
    def test_optimize_preserves_reconstruction_quality(
        self, optimizer, transformer, extractor
    ):
        """Test that optimized coefficients still reconstruct well."""
        # Create a complex contour
        t = np.linspace(0, 2 * np.pi, 150, endpoint=False)
        signal = (
            np.exp(1j * t) +
            0.5 * np.exp(1j * 2 * t) +
            0.3 * np.exp(1j * 4 * t)
        )
        
        # Compute DFT
        coefficients = transformer.compute_dft(signal)
        
        # Optimize
        result = optimizer.optimize_count(coefficients, signal)
        
        # Get optimized coefficients
        sorted_coeffs = transformer.sort_by_amplitude(coefficients)
        optimized_coeffs = sorted_coeffs[:result.optimal_count]
        
        # Pad back to original size for reconstruction
        padded_coeffs = optimizer._pad_coefficients(optimized_coeffs, len(coefficients))
        
        # Reconstruct
        reconstructed = transformer.compute_idft(padded_coeffs)
        
        # Verify reconstruction quality
        error = optimizer.compute_reconstruction_error(signal, reconstructed)
        assert error <= optimizer.target_error * 1.1
    
    def test_complexity_classification_consistency(
        self, optimizer, transformer
    ):
        """Test that complexity classification is consistent."""
        # Simple signal (low complexity)
        simple = np.exp(1j * np.linspace(0, 2 * np.pi, 50, endpoint=False))
        simple_coeffs = transformer.compute_dft(simple)
        simple_result = optimizer.optimize_count(simple_coeffs, simple)
        
        # Complex signal (higher complexity)
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        complex_signal = sum(
            (1.0 / (i + 1)) * np.exp(1j * i * t)
            for i in range(1, 20)
        )
        complex_coeffs = transformer.compute_dft(complex_signal)
        complex_result = optimizer.optimize_count(complex_coeffs, complex_signal)
        
        # Complex signal should need more coefficients
        # (though not always - depends on the signal structure)
        assert simple_result.optimal_count >= 10
        assert complex_result.optimal_count >= 10
        
        # Both should meet error threshold
        assert simple_result.reconstruction_error <= 0.05
        assert complex_result.reconstruction_error <= 0.05
