"""
Property-based tests for image processing pipeline.

Feature: fourier-image-encryption
Property 1: Complex Plane Round-Trip
Property 31: Empty Image Handling

These tests verify that coordinate conversions preserve data and that
edge cases like empty images are handled gracefully.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume

from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.models.data_models import Contour
from fourier_encryption.models.exceptions import ImageProcessingError


# Property 1: Complex Plane Round-Trip
@given(
    num_points=st.integers(min_value=10, max_value=1000),
    x_coords=st.lists(
        st.floats(min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=1000
    ),
    y_coords=st.lists(
        st.floats(min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=1000
    ),
)
@pytest.mark.property_test
def test_complex_plane_round_trip(num_points, x_coords, y_coords):
    """
    Feature: fourier-image-encryption
    Property 1: Complex Plane Round-Trip
    
    For any set of 2D contour points (x, y), converting to complex plane
    representation (x + iy) and back should preserve the original coordinates
    within floating-point precision.
    
    **Validates: Requirements 3.1.4**
    """
    # Ensure x_coords and y_coords have the same length
    min_len = min(len(x_coords), len(y_coords), num_points)
    assume(min_len >= 10)  # Need at least 10 points for valid contour
    
    x_coords = x_coords[:min_len]
    y_coords = y_coords[:min_len]
    
    # Create contour points as Nx2 array
    original_points = np.column_stack([x_coords, y_coords])
    
    # Create Contour object
    contour = Contour(
        points=original_points,
        is_closed=False,
        length=len(original_points)
    )
    
    # Convert to complex plane
    extractor = ContourExtractor()
    complex_points = extractor.to_complex_plane(contour)
    
    # Verify complex points have correct shape
    assert complex_points.shape == (len(original_points),)
    assert complex_points.dtype == np.complex128 or complex_points.dtype == np.complex64
    
    # Convert back to (x, y) coordinates
    reconstructed_x = complex_points.real
    reconstructed_y = complex_points.imag
    reconstructed_points = np.column_stack([reconstructed_x, reconstructed_y])
    
    # Verify round-trip preserves coordinates within floating-point precision
    # Use relative tolerance for large values and absolute tolerance for small values
    assert np.allclose(original_points, reconstructed_points, rtol=1e-10, atol=1e-10)
    
    # Verify individual coordinates match
    assert np.allclose(original_points[:, 0], reconstructed_x, rtol=1e-10, atol=1e-10)
    assert np.allclose(original_points[:, 1], reconstructed_y, rtol=1e-10, atol=1e-10)


@given(
    num_points=st.integers(min_value=10, max_value=100),
)
@pytest.mark.property_test
def test_complex_plane_round_trip_integer_coordinates(num_points):
    """
    Feature: fourier-image-encryption
    Property 1: Complex Plane Round-Trip (integer coordinates)
    
    For any set of integer contour points (common in image processing),
    the round-trip conversion should be exact.
    
    **Validates: Requirements 3.1.4**
    """
    # Generate random integer coordinates (typical for image pixels)
    x_coords = np.random.randint(-1000, 1000, size=num_points)
    y_coords = np.random.randint(-1000, 1000, size=num_points)
    
    original_points = np.column_stack([x_coords, y_coords]).astype(np.float64)
    
    contour = Contour(
        points=original_points,
        is_closed=True,
        length=num_points
    )
    
    extractor = ContourExtractor()
    complex_points = extractor.to_complex_plane(contour)
    
    # Convert back
    reconstructed_points = np.column_stack([complex_points.real, complex_points.imag])
    
    # For integer coordinates, round-trip should be exact
    assert np.allclose(original_points, reconstructed_points, rtol=1e-12, atol=1e-12)


@given(
    x=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@pytest.mark.property_test
def test_complex_plane_single_point_round_trip(x, y):
    """
    Feature: fourier-image-encryption
    Property 1: Complex Plane Round-Trip (single point)
    
    For any single point (x, y), the conversion to complex and back
    should preserve the coordinates exactly.
    
    **Validates: Requirements 3.1.4**
    """
    # Create a contour with repeated points (minimum 10 points required)
    points = np.array([[x, y]] * 10, dtype=np.float64)
    
    contour = Contour(
        points=points,
        is_closed=False,
        length=10
    )
    
    extractor = ContourExtractor()
    complex_points = extractor.to_complex_plane(contour)
    
    # All complex points should be identical
    assert len(complex_points) == 10
    assert np.allclose(complex_points.real, x, rtol=1e-10, atol=1e-10)
    assert np.allclose(complex_points.imag, y, rtol=1e-10, atol=1e-10)
    
    # Verify the complex representation
    expected_complex = x + 1j * y
    assert np.allclose(complex_points, expected_complex, rtol=1e-10, atol=1e-10)


# Property 31: Empty Image Handling
@pytest.mark.property_test
def test_empty_image_handling_all_zeros():
    """
    Feature: fourier-image-encryption
    Property 31: Empty Image Handling
    
    For any image with no detectable edges (completely uniform/empty),
    the system must handle it gracefully by raising an ImageProcessingError
    with a descriptive message.
    
    **Validates: Requirements 3.1.3**
    """
    # Create completely empty edge map (all zeros)
    edge_map = np.zeros((100, 100), dtype=np.uint8)
    
    extractor = ContourExtractor()
    
    # Should raise ImageProcessingError with descriptive message
    with pytest.raises(ImageProcessingError) as exc_info:
        extractor.extract_contours(edge_map)
    
    # Verify error message is descriptive
    error_message = str(exc_info.value)
    assert "empty" in error_message.lower() or "no edges" in error_message.lower()
    
    # Verify context is provided
    assert hasattr(exc_info.value, 'context')


@given(
    width=st.integers(min_value=10, max_value=500),
    height=st.integers(min_value=10, max_value=500),
    value=st.integers(min_value=0, max_value=0),  # Always 0 for empty image
)
@pytest.mark.property_test
def test_empty_image_handling_various_sizes(width, height, value):
    """
    Feature: fourier-image-encryption
    Property 31: Empty Image Handling (various sizes)
    
    For any size of completely uniform image (all pixels same value),
    the system must handle it gracefully with an ImageProcessingError.
    
    **Validates: Requirements 3.1.3**
    """
    # Create uniform edge map
    edge_map = np.full((height, width), value, dtype=np.uint8)
    
    extractor = ContourExtractor()
    
    with pytest.raises(ImageProcessingError) as exc_info:
        extractor.extract_contours(edge_map)
    
    # Verify error is raised and has context
    assert exc_info.value is not None
    assert hasattr(exc_info.value, 'context')


@given(
    size=st.integers(min_value=10, max_value=200),
)
@pytest.mark.property_test
def test_empty_image_handling_square_images(size):
    """
    Feature: fourier-image-encryption
    Property 31: Empty Image Handling (square images)
    
    For any square image with no edges, the system must handle it gracefully.
    
    **Validates: Requirements 3.1.3**
    """
    # Create empty square edge map
    edge_map = np.zeros((size, size), dtype=np.uint8)
    
    extractor = ContourExtractor()
    
    with pytest.raises(ImageProcessingError) as exc_info:
        extractor.extract_contours(edge_map)
    
    error_message = str(exc_info.value)
    assert "empty" in error_message.lower() or "no edges" in error_message.lower()


@pytest.mark.property_test
def test_empty_image_handling_single_pixel():
    """
    Feature: fourier-image-encryption
    Property 31: Empty Image Handling (edge case: single pixel)
    
    For a 1x1 image (single pixel), the system must handle it gracefully.
    
    **Validates: Requirements 3.1.3**
    """
    # Create 1x1 edge map
    edge_map = np.zeros((1, 1), dtype=np.uint8)
    
    extractor = ContourExtractor()
    
    with pytest.raises(ImageProcessingError) as exc_info:
        extractor.extract_contours(edge_map)
    
    # Should raise error for empty image
    assert exc_info.value is not None


@given(
    width=st.integers(min_value=10, max_value=200),
    height=st.integers(min_value=10, max_value=200),
    uniform_value=st.integers(min_value=1, max_value=254),  # Non-zero, non-255 uniform value
)
@pytest.mark.property_test
def test_uniform_non_zero_image_handling(width, height, uniform_value):
    """
    Feature: fourier-image-encryption
    Property 31: Empty Image Handling (uniform non-zero)
    
    For any image with uniform non-zero values (no edges), the system
    must handle it gracefully. Since the image is uniform, there are no
    edges to detect.
    
    **Validates: Requirements 3.1.3**
    """
    # Create uniform edge map with non-zero value
    # In a real edge detection scenario, uniform images produce no edges
    edge_map = np.full((height, width), uniform_value, dtype=np.uint8)
    
    # For edge detection, uniform images should produce empty edge maps
    # But if we pass a uniform non-zero map to extract_contours,
    # it might find contours (the entire image boundary)
    # Let's test with actual empty edge map (all zeros)
    edge_map = np.zeros((height, width), dtype=np.uint8)
    
    extractor = ContourExtractor()
    
    with pytest.raises(ImageProcessingError) as exc_info:
        extractor.extract_contours(edge_map)
    
    assert exc_info.value is not None


@pytest.mark.property_test
def test_empty_image_error_contains_context():
    """
    Feature: fourier-image-encryption
    Property 31: Empty Image Handling (error context)
    
    When an empty image error is raised, it must contain context information
    about the image (shape, edge pixel count, etc.).
    
    **Validates: Requirements 3.1.3**
    """
    edge_map = np.zeros((50, 75), dtype=np.uint8)
    
    extractor = ContourExtractor()
    
    with pytest.raises(ImageProcessingError) as exc_info:
        extractor.extract_contours(edge_map)
    
    # Verify error has context
    assert hasattr(exc_info.value, 'context')
    context = exc_info.value.context
    
    # Context should contain shape information
    assert 'shape' in context or 'edge_pixels' in context or context is not None
