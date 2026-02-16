"""
Unit tests for edge detection strategies.

Tests Canny edge detector on sample images, validates binary output,
and verifies performance metrics collection.
"""

import numpy as np
import pytest

from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.models.exceptions import ImageProcessingError


class TestCannyEdgeDetector:
    """Tests for CannyEdgeDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a CannyEdgeDetector instance with default settings."""
        return CannyEdgeDetector()
    
    @pytest.fixture
    def detector_fixed_threshold(self):
        """Create a CannyEdgeDetector with fixed thresholds (no adaptive)."""
        return CannyEdgeDetector(
            low_threshold=50,
            high_threshold=150,
            use_adaptive_threshold=False
        )
    
    @pytest.fixture
    def sample_gradient_image(self):
        """Create a sample grayscale image with gradient."""
        # Create a 100x100 grayscale image with horizontal gradient
        image = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            image[:, i] = i * 2  # Gradient from 0 to ~200
        return image
    
    @pytest.fixture
    def sample_square_image(self):
        """Create a sample image with a white square on black background."""
        image = np.zeros((100, 100), dtype=np.uint8)
        # Draw a white square in the center
        image[25:75, 25:75] = 255
        return image
    
    @pytest.fixture
    def sample_circle_image(self):
        """Create a sample image with a white circle on black background."""
        image = np.zeros((100, 100), dtype=np.uint8)
        # Draw a circle
        center = (50, 50)
        radius = 30
        y, x = np.ogrid[:100, :100]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = 255
        return image
    
    def test_detector_initialization_default(self):
        """Test that detector initializes with default parameters."""
        detector = CannyEdgeDetector()
        
        assert detector.low_threshold == 50
        assert detector.high_threshold == 150
        assert detector.aperture_size == 3
        assert detector.use_adaptive_threshold is True
    
    def test_detector_initialization_custom(self):
        """Test that detector initializes with custom parameters."""
        detector = CannyEdgeDetector(
            low_threshold=30,
            high_threshold=100,
            aperture_size=5,
            use_adaptive_threshold=False
        )
        
        assert detector.low_threshold == 30
        assert detector.high_threshold == 100
        assert detector.aperture_size == 5
        assert detector.use_adaptive_threshold is False
    
    def test_detector_invalid_aperture_size_even(self):
        """Test that even aperture size raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CannyEdgeDetector(aperture_size=4)
        assert "must be odd" in str(exc_info.value)
    
    def test_detector_invalid_aperture_size_too_small(self):
        """Test that aperture size < 3 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CannyEdgeDetector(aperture_size=1)
        assert "must be odd" in str(exc_info.value)
    
    def test_detector_invalid_aperture_size_too_large(self):
        """Test that aperture size > 7 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CannyEdgeDetector(aperture_size=9)
        assert "must be odd" in str(exc_info.value)
    
    def test_detect_edges_gradient_image(self, detector, sample_gradient_image):
        """Test Canny detector on gradient image."""
        edges = detector.detect_edges(sample_gradient_image)
        
        # Should return a valid edge map
        assert edges is not None
        assert isinstance(edges, np.ndarray)
        assert edges.shape == sample_gradient_image.shape
        assert edges.dtype == np.uint8
    
    def test_detect_edges_square_image(self, detector, sample_square_image):
        """Test Canny detector on image with square."""
        edges = detector.detect_edges(sample_square_image)
        
        # Should detect edges of the square
        assert edges is not None
        assert edges.shape == sample_square_image.shape
        # Should have some edge pixels
        assert np.count_nonzero(edges) > 0
    
    def test_detect_edges_circle_image(self, detector, sample_circle_image):
        """Test Canny detector on image with circle."""
        edges = detector.detect_edges(sample_circle_image)
        
        # Should detect edges of the circle
        assert edges is not None
        assert edges.shape == sample_circle_image.shape
        # Should have some edge pixels
        assert np.count_nonzero(edges) > 0
    
    def test_detect_edges_output_is_binary(self, detector, sample_square_image):
        """Test that output is binary (0 or 255 only)."""
        edges = detector.detect_edges(sample_square_image)
        
        # Check that all values are either 0 or 255
        unique_values = np.unique(edges)
        assert len(unique_values) <= 2
        assert all(val in [0, 255] for val in unique_values)
    
    def test_detect_edges_binary_gradient_image(self, detector, sample_gradient_image):
        """Test that output is binary for gradient image."""
        edges = detector.detect_edges(sample_gradient_image)
        
        # Check that all values are either 0 or 255
        unique_values = np.unique(edges)
        assert len(unique_values) <= 2
        assert all(val in [0, 255] for val in unique_values)
    
    def test_detect_edges_binary_circle_image(self, detector, sample_circle_image):
        """Test that output is binary for circle image."""
        edges = detector.detect_edges(sample_circle_image)
        
        # Check that all values are either 0 or 255
        unique_values = np.unique(edges)
        assert len(unique_values) <= 2
        assert all(val in [0, 255] for val in unique_values)
    
    def test_detect_edges_fixed_threshold(self, detector_fixed_threshold, sample_square_image):
        """Test edge detection with fixed thresholds (no adaptive)."""
        edges = detector_fixed_threshold.detect_edges(sample_square_image)
        
        assert edges is not None
        assert edges.shape == sample_square_image.shape
        # Should be binary
        unique_values = np.unique(edges)
        assert all(val in [0, 255] for val in unique_values)
    
    def test_detect_edges_normalized_float_input(self, detector):
        """Test edge detection with normalized float input [0, 1]."""
        # Create normalized float image
        image = np.random.rand(100, 100).astype(np.float32)
        
        edges = detector.detect_edges(image)
        
        assert edges is not None
        assert edges.shape == image.shape
        assert edges.dtype == np.uint8
        # Should be binary
        unique_values = np.unique(edges)
        assert all(val in [0, 255] for val in unique_values)
    
    def test_detect_edges_float_input_large_values(self, detector):
        """Test edge detection with float input with values > 1."""
        # Create float image with values in [0, 255] range
        image = np.random.rand(100, 100).astype(np.float32) * 255
        
        edges = detector.detect_edges(image)
        
        assert edges is not None
        assert edges.shape == image.shape
        assert edges.dtype == np.uint8
    
    def test_detect_edges_empty_image_raises_error(self, detector):
        """Test that empty image raises ImageProcessingError."""
        empty_image = np.array([], dtype=np.uint8)
        
        with pytest.raises(ImageProcessingError) as exc_info:
            detector.detect_edges(empty_image)
        assert "empty" in str(exc_info.value).lower()
    
    def test_detect_edges_invalid_input_type(self, detector):
        """Test that non-array input raises ImageProcessingError."""
        with pytest.raises(ImageProcessingError) as exc_info:
            detector.detect_edges("not an array")
        assert "NumPy array" in str(exc_info.value)
    
    def test_detect_edges_3d_image_raises_error(self, detector):
        """Test that 3D (color) image raises ImageProcessingError."""
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ImageProcessingError) as exc_info:
            detector.detect_edges(color_image)
        assert "grayscale" in str(exc_info.value).lower()
    
    def test_performance_metrics_collected(self, detector, sample_square_image):
        """Test that performance metrics are collected after edge detection."""
        # Perform edge detection
        edges = detector.detect_edges(sample_square_image)
        
        # Get metrics
        metrics = detector.get_performance_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "detection_time" in metrics
        assert "edge_density" in metrics
        assert "image_size" in metrics
    
    def test_performance_metrics_detection_time(self, detector, sample_square_image):
        """Test that detection_time metric is valid."""
        detector.detect_edges(sample_square_image)
        metrics = detector.get_performance_metrics()
        
        # Detection time should be non-negative
        assert metrics["detection_time"] >= 0
        # Should be reasonable (less than 1 second for small image)
        assert metrics["detection_time"] < 1.0
    
    def test_performance_metrics_edge_density(self, detector, sample_square_image):
        """Test that edge_density metric is valid."""
        detector.detect_edges(sample_square_image)
        metrics = detector.get_performance_metrics()
        
        # Edge density should be in [0, 100] range
        assert 0 <= metrics["edge_density"] <= 100
        # For square image, should have some edges
        assert metrics["edge_density"] > 0
    
    def test_performance_metrics_image_size(self, detector, sample_square_image):
        """Test that image_size metric is correct."""
        detector.detect_edges(sample_square_image)
        metrics = detector.get_performance_metrics()
        
        # Image size should match input size
        expected_size = sample_square_image.size
        assert metrics["image_size"] == float(expected_size)
    
    def test_performance_metrics_updated_each_call(self, detector, sample_square_image):
        """Test that metrics are updated for each detection call."""
        # First detection
        detector.detect_edges(sample_square_image)
        metrics1 = detector.get_performance_metrics()
        
        # Create different image
        different_image = np.zeros((50, 50), dtype=np.uint8)
        different_image[10:40, 10:40] = 255
        
        # Second detection
        detector.detect_edges(different_image)
        metrics2 = detector.get_performance_metrics()
        
        # Image size should be different
        assert metrics1["image_size"] != metrics2["image_size"]
        assert metrics2["image_size"] == float(different_image.size)
    
    def test_performance_metrics_zero_edges(self, detector):
        """Test metrics when no edges are detected."""
        # Create uniform image (no edges)
        uniform_image = np.full((100, 100), 128, dtype=np.uint8)
        
        detector.detect_edges(uniform_image)
        metrics = detector.get_performance_metrics()
        
        # Edge density should be 0 or very low
        assert metrics["edge_density"] >= 0
        # Other metrics should still be valid
        assert metrics["detection_time"] >= 0
        assert metrics["image_size"] == float(uniform_image.size)
    
    def test_detect_edges_different_aperture_sizes(self, sample_square_image):
        """Test edge detection with different valid aperture sizes."""
        valid_apertures = [3, 5, 7]
        
        for aperture in valid_apertures:
            detector = CannyEdgeDetector(aperture_size=aperture)
            edges = detector.detect_edges(sample_square_image)
            
            assert edges is not None
            assert edges.shape == sample_square_image.shape
            # Should be binary
            unique_values = np.unique(edges)
            assert all(val in [0, 255] for val in unique_values)
    
    def test_detect_edges_adaptive_vs_fixed_threshold(self, sample_square_image):
        """Test that adaptive and fixed thresholds both produce valid output."""
        # Adaptive threshold
        detector_adaptive = CannyEdgeDetector(use_adaptive_threshold=True)
        edges_adaptive = detector_adaptive.detect_edges(sample_square_image)
        
        # Fixed threshold
        detector_fixed = CannyEdgeDetector(use_adaptive_threshold=False)
        edges_fixed = detector_fixed.detect_edges(sample_square_image)
        
        # Both should produce valid binary edge maps
        assert edges_adaptive.shape == sample_square_image.shape
        assert edges_fixed.shape == sample_square_image.shape
        assert all(val in [0, 255] for val in np.unique(edges_adaptive))
        assert all(val in [0, 255] for val in np.unique(edges_fixed))
