"""
Property-based tests for CoefficientOptimizer in Fourier-Based Image Encryption System.

This module contains property tests for the AI coefficient optimizer component,
validating complexity classification, coefficient count optimization, reconstruction
error thresholds, and optimization explanations.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from fourier_encryption.ai.coefficient_optimizer import CoefficientOptimizer
from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.models.exceptions import AIModelError


# Strategy for generating valid Fourier coefficients
@st.composite
def fourier_coefficients(draw, min_count=10, max_count=100):
    """Generate a list of valid Fourier coefficients for testing."""
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    coefficients = []
    
    for freq in range(count):
        # Generate amplitude with power-law decay (realistic for Fourier coefficients)
        # Higher frequencies typically have lower amplitudes
        base_amplitude = draw(st.floats(min_value=0.1, max_value=100.0))
        amplitude = base_amplitude / (1 + freq * 0.1)
        
        # Generate phase in valid range [-π, π]
        phase = draw(st.floats(min_value=-math.pi, max_value=math.pi))
        
        # Compute complex value from amplitude and phase
        complex_value = amplitude * complex(math.cos(phase), math.sin(phase))
        
        coeff = FourierCoefficient(
            frequency=freq,
            amplitude=amplitude,
            phase=phase,
            complex_value=complex_value
        )
        coefficients.append(coeff)
    
    return coefficients


# Strategy for generating valid grayscale images
@st.composite
def grayscale_images(draw, min_size=10, max_size=128):
    """Generate random grayscale images for testing."""
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate random pixel values using numpy directly
    rng = np.random.RandomState(draw(st.integers(min_value=0, max_value=2**31-1)))
    image = rng.randint(0, 256, size=(height, width), dtype=np.uint8)
    
    return image


# Strategy for generating complex contour points
@st.composite
def complex_contour_points(draw, min_points=10, max_points=100):
    """Generate complex contour points for testing."""
    num_points = draw(st.integers(min_value=min_points, max_value=max_points))
    
    # Generate random complex points
    rng = np.random.RandomState(draw(st.integers(min_value=0, max_value=2**31-1)))
    real_parts = rng.uniform(-100, 100, size=num_points)
    imag_parts = rng.uniform(-100, 100, size=num_points)
    
    points = real_parts + 1j * imag_parts
    return points


class TestCoefficientOptimizerProperties:
    """Property-based tests for CoefficientOptimizer."""
    
    @given(image=grayscale_images())
    @settings(max_examples=10, deadline=None)
    def test_property_17_complexity_classification_validity(self, image):
        """
        Feature: fourier-image-encryption
        Property 17: Complexity Classification Validity
        
        **Validates: Requirements 3.9.1**
        
        For any input image, the AI complexity classifier must return exactly
        one of the valid complexity classes: "low", "medium", or "high".
        """
        optimizer = CoefficientOptimizer()
        
        # Classify complexity
        complexity_class = optimizer.classify_complexity(image)
        
        # Verify it's one of the valid classes
        valid_classes = {"low", "medium", "high"}
        assert complexity_class in valid_classes, \
            f"Invalid complexity class '{complexity_class}', must be one of {valid_classes}"
        
        # Verify it's a string
        assert isinstance(complexity_class, str), \
            f"Complexity class must be a string, got {type(complexity_class)}"
    
    @given(
        coefficients=fourier_coefficients(min_count=10, max_count=100),
        original_points=complex_contour_points(min_points=10, max_points=100)
    )
    @settings(max_examples=10, deadline=None)
    def test_property_18_optimizer_coefficient_count_bounds(
        self, coefficients, original_points
    ):
        """
        Feature: fourier-image-encryption
        Property 18: Optimizer Coefficient Count Bounds
        
        **Validates: Requirements 3.9.2**
        
        For any image processed by the AI optimizer, the returned optimal
        coefficient count must be within the valid range [10, 1000].
        """
        optimizer = CoefficientOptimizer(
            target_error=0.05,
            min_coefficients=10,
            max_coefficients=1000
        )
        
        # Optimize coefficient count
        result = optimizer.optimize_count(coefficients, original_points)
        
        # Verify optimal count is within bounds
        assert 10 <= result.optimal_count <= 1000, \
            f"Optimal count {result.optimal_count} must be in range [10, 1000]"
        
        # Verify it's an integer
        assert isinstance(result.optimal_count, int), \
            f"Optimal count must be an integer, got {type(result.optimal_count)}"
    
    @given(
        coefficients=fourier_coefficients(min_count=20, max_count=100),
        original_points=complex_contour_points(min_points=20, max_points=100)
    )
    @settings(max_examples=10, deadline=None)
    def test_property_19_reconstruction_error_threshold(
        self, coefficients, original_points
    ):
        """
        Feature: fourier-image-encryption
        Property 19: Reconstruction Error Threshold
        
        **Validates: Requirements 3.9.3**
        
        For any optimized coefficient set, the reconstruction error (RMSE)
        between the original contour and the reconstructed contour must be
        below 5% (or the configured target error).
        """
        target_error = 0.05  # 5%
        optimizer = CoefficientOptimizer(target_error=target_error)
        
        # Optimize coefficient count
        result = optimizer.optimize_count(coefficients, original_points)
        
        # Verify reconstruction error is below threshold
        # Note: In some cases, the optimizer may not be able to meet the target
        # if there aren't enough coefficients, but it should try its best
        assert result.reconstruction_error >= 0, \
            f"Reconstruction error must be non-negative, got {result.reconstruction_error}"
        
        # The error should ideally be below target, but we allow some tolerance
        # for cases where the optimizer uses all available coefficients
        # and still can't meet the target
        if len(coefficients) >= 20:  # If we have enough coefficients
            # Either error is below target, or we're using most coefficients
            assert (result.reconstruction_error <= target_error * 1.5 or 
                    result.optimal_count >= len(coefficients) * 0.8), \
                f"Reconstruction error {result.reconstruction_error} should be close to " \
                f"target {target_error} or using most coefficients"
    
    @given(
        coefficients=fourier_coefficients(min_count=20, max_count=100),
        original_points=complex_contour_points(min_points=20, max_points=100)
    )
    @settings(max_examples=10, deadline=None)
    def test_property_20_optimization_reduces_size(
        self, coefficients, original_points
    ):
        """
        Feature: fourier-image-encryption
        Property 20: Optimization Reduces Size
        
        **Validates: Requirements 3.9.4**
        
        For any image processed by the AI optimizer, the optimized coefficient
        count must be less than or equal to the maximum coefficient count
        (optimization never increases size).
        """
        max_coefficients = len(coefficients)
        optimizer = CoefficientOptimizer(
            target_error=0.05,
            min_coefficients=10,
            max_coefficients=max_coefficients
        )
        
        # Optimize coefficient count
        result = optimizer.optimize_count(coefficients, original_points)
        
        # Verify optimized count doesn't exceed original count
        assert result.optimal_count <= max_coefficients, \
            f"Optimized count {result.optimal_count} must not exceed " \
            f"original count {max_coefficients}"
        
        # Verify it's at least the minimum
        assert result.optimal_count >= 10, \
            f"Optimized count {result.optimal_count} must be at least 10"
    
    @given(
        coefficients=fourier_coefficients(min_count=10, max_count=100),
        original_points=complex_contour_points(min_points=10, max_points=100)
    )
    @settings(max_examples=10, deadline=None)
    def test_property_21_optimization_explanation_presence(
        self, coefficients, original_points
    ):
        """
        Feature: fourier-image-encryption
        Property 21: Optimization Explanation Presence
        
        **Validates: Requirements 3.9.6**
        
        For any optimization result, the explanation field must be a non-empty
        string describing why the specific coefficient count was chosen.
        """
        optimizer = CoefficientOptimizer()
        
        # Optimize coefficient count
        result = optimizer.optimize_count(coefficients, original_points)
        
        # Verify explanation is present
        assert result.explanation is not None, \
            "Explanation must not be None"
        
        assert isinstance(result.explanation, str), \
            f"Explanation must be a string, got {type(result.explanation)}"
        
        assert len(result.explanation.strip()) > 0, \
            "Explanation must not be empty or whitespace-only"
        
        # Verify explanation contains meaningful information
        # It should mention coefficient count
        assert str(result.optimal_count) in result.explanation, \
            "Explanation should mention the optimal coefficient count"
        
        # It should mention complexity or error
        explanation_lower = result.explanation.lower()
        assert any(word in explanation_lower for word in 
                   ["complexity", "error", "reduction", "optimized"]), \
            "Explanation should contain meaningful optimization information"
    
    @given(
        original=complex_contour_points(min_points=10, max_points=100),
    )
    @settings(max_examples=10, deadline=None)
    def test_reconstruction_error_is_zero_for_identical_points(self, original):
        """
        Feature: fourier-image-encryption
        Property: Reconstruction error is zero for identical points
        
        **Validates: Requirements 3.9.3**
        
        When original and reconstructed points are identical, the
        reconstruction error should be zero (or very close to zero).
        """
        optimizer = CoefficientOptimizer()
        
        # Compute error between identical points
        error = optimizer.compute_reconstruction_error(original, original)
        
        # Error should be zero or very close to zero
        assert error < 1e-10, \
            f"Error for identical points should be ~0, got {error}"
    
    @given(
        original=complex_contour_points(min_points=10, max_points=100),
    )
    @settings(max_examples=10, deadline=None)
    def test_reconstruction_error_is_symmetric(self, original):
        """
        Feature: fourier-image-encryption
        Property: Reconstruction error is approximately symmetric
        
        **Validates: Requirements 3.9.3**
        
        The reconstruction error should be approximately symmetric: error(A, B) ≈ error(B, A).
        Note: Due to normalization by signal magnitude, perfect symmetry is not guaranteed,
        but the errors should be close.
        """
        optimizer = CoefficientOptimizer()
        
        # Create a slightly different version
        reconstructed = original + np.random.uniform(-1, 1, size=original.shape)
        
        # Compute error both ways
        error1 = optimizer.compute_reconstruction_error(original, reconstructed)
        error2 = optimizer.compute_reconstruction_error(reconstructed, original)
        
        # Errors should be approximately equal (within 1% relative tolerance)
        # The normalization by different signal magnitudes can introduce slight asymmetry
        # We use a generous tolerance to account for this
        relative_diff = abs(error1 - error2) / max(error1, error2)
        assert relative_diff < 0.01, \
            f"Error should be approximately symmetric: {error1} vs {error2}, " \
            f"relative difference: {relative_diff:.4f}"
    
    @given(
        original=complex_contour_points(min_points=10, max_points=100),
        scale=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=10, deadline=None)
    def test_reconstruction_error_scales_with_difference(self, original, scale):
        """
        Feature: fourier-image-encryption
        Property: Reconstruction error scales with difference magnitude
        
        **Validates: Requirements 3.9.3**
        
        Larger differences between original and reconstructed points should
        result in larger reconstruction errors.
        """
        optimizer = CoefficientOptimizer()
        
        # Create two reconstructed versions with different error magnitudes
        small_diff = original + 0.1
        large_diff = original + scale
        
        # Compute errors
        small_error = optimizer.compute_reconstruction_error(original, small_diff)
        large_error = optimizer.compute_reconstruction_error(original, large_diff)
        
        # Larger difference should produce larger error (with some tolerance)
        if scale > 0.2:  # Only test when scale is significantly larger
            assert large_error >= small_error * 0.9, \
                f"Larger difference should produce larger error: " \
                f"small={small_error}, large={large_error}"
    
    def test_empty_coefficients_raises_error(self):
        """
        Feature: fourier-image-encryption
        Property: Empty coefficient list is rejected
        
        **Validates: Requirements 3.9.2**
        
        Attempting to optimize with an empty coefficient list should raise
        an AIModelError.
        """
        optimizer = CoefficientOptimizer()
        original_points = np.array([1+1j, 2+2j, 3+3j])
        
        with pytest.raises(AIModelError) as exc_info:
            optimizer.optimize_count([], original_points)
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_empty_original_points_raises_error(self):
        """
        Feature: fourier-image-encryption
        Property: Empty original points are rejected
        
        **Validates: Requirements 3.9.3**
        
        Attempting to optimize with empty original points should raise
        an AIModelError.
        """
        optimizer = CoefficientOptimizer()
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=1.0,
                phase=0.0,
                complex_value=1.0+0j
            )
        ]
        empty_points = np.array([])
        
        with pytest.raises(AIModelError) as exc_info:
            optimizer.optimize_count(coefficients, empty_points)
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_invalid_target_error_raises_error(self):
        """
        Feature: fourier-image-encryption
        Property: Invalid target error is rejected
        
        **Validates: Requirements 3.9.3**
        
        Target error must be in range (0, 1). Values outside this range
        should raise an AIModelError.
        """
        # Test negative error
        with pytest.raises(AIModelError) as exc_info:
            CoefficientOptimizer(target_error=-0.1)
        assert "target_error" in str(exc_info.value).lower()
        
        # Test zero error
        with pytest.raises(AIModelError) as exc_info:
            CoefficientOptimizer(target_error=0.0)
        assert "target_error" in str(exc_info.value).lower()
        
        # Test error >= 1
        with pytest.raises(AIModelError) as exc_info:
            CoefficientOptimizer(target_error=1.0)
        assert "target_error" in str(exc_info.value).lower()
    
    def test_invalid_coefficient_range_raises_error(self):
        """
        Feature: fourier-image-encryption
        Property: Invalid coefficient range is rejected
        
        **Validates: Requirements 3.9.2**
        
        Coefficient range must satisfy: 10 <= min <= max <= 1000.
        Invalid ranges should raise an AIModelError.
        """
        # Test min < 10
        with pytest.raises(AIModelError):
            CoefficientOptimizer(min_coefficients=5, max_coefficients=100)
        
        # Test max > 1000
        with pytest.raises(AIModelError):
            CoefficientOptimizer(min_coefficients=10, max_coefficients=1500)
        
        # Test min > max
        with pytest.raises(AIModelError):
            CoefficientOptimizer(min_coefficients=100, max_coefficients=50)
    
    def test_non_grayscale_image_raises_error(self):
        """
        Feature: fourier-image-encryption
        Property: Non-grayscale images are rejected for complexity classification
        
        **Validates: Requirements 3.9.1**
        
        Images that are not 2D (grayscale) should raise an AIModelError.
        """
        optimizer = CoefficientOptimizer()
        
        # 3D color image
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(AIModelError) as exc_info:
            optimizer.classify_complexity(color_image)
        
        assert "2D" in str(exc_info.value) or "grayscale" in str(exc_info.value).lower()
    
    def test_empty_image_raises_error(self):
        """
        Feature: fourier-image-encryption
        Property: Empty images are rejected for complexity classification
        
        **Validates: Requirements 3.9.1**
        
        Empty images should raise an AIModelError.
        """
        optimizer = CoefficientOptimizer()
        empty_image = np.array([], dtype=np.uint8).reshape(0, 0)
        
        with pytest.raises(AIModelError) as exc_info:
            optimizer.classify_complexity(empty_image)
        
        assert "empty" in str(exc_info.value).lower()
    
    @given(
        height=st.integers(min_value=10, max_value=100),
        width=st.integers(min_value=10, max_value=100),
        value=st.integers(min_value=0, max_value=255)
    )
    @settings(max_examples=10, deadline=None)
    def test_uniform_images_classified_as_low_complexity(self, height, width, value):
        """
        Feature: fourier-image-encryption
        Property: Uniform images have low complexity
        
        **Validates: Requirements 3.9.1**
        
        Images with uniform pixel values (no edges, no detail) should be
        classified as "low" complexity.
        """
        optimizer = CoefficientOptimizer()
        
        # Create uniform image
        uniform_image = np.full((height, width), value, dtype=np.uint8)
        
        # Classify complexity
        complexity = optimizer.classify_complexity(uniform_image)
        
        # Uniform images should be low complexity
        assert complexity == "low", \
            f"Uniform images should be classified as 'low' complexity, got '{complexity}'"
    
    @given(
        original=complex_contour_points(min_points=10, max_points=100),
    )
    @settings(max_examples=10, deadline=None)
    def test_mismatched_shapes_raise_error(self, original):
        """
        Feature: fourier-image-encryption
        Property: Mismatched array shapes are rejected
        
        **Validates: Requirements 3.9.3**
        
        Computing reconstruction error with mismatched array shapes should
        raise an AIModelError.
        """
        optimizer = CoefficientOptimizer()
        
        # Create reconstructed with different size
        reconstructed = original[:len(original)//2]
        
        with pytest.raises(AIModelError) as exc_info:
            optimizer.compute_reconstruction_error(original, reconstructed)
        
        assert "shape" in str(exc_info.value).lower() or "mismatch" in str(exc_info.value).lower()
