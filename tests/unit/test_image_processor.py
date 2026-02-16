"""
Unit tests for image processing pipeline.

Tests image loading, grayscale conversion, preprocessing operations,
and edge case handling for the ImageProcessor and ContourExtractor classes.
"""

import numpy as np
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
import cv2

from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.config.settings import PreprocessConfig
from fourier_encryption.models.exceptions import ImageProcessingError
from fourier_encryption.models.data_models import Contour


class TestOpenCVImageProcessor:
    """Tests for OpenCVImageProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create an OpenCVImageProcessor instance."""
        return OpenCVImageProcessor()
    
    @pytest.fixture
    def sample_color_image(self):
        """Create a sample color image (BGR format)."""
        # Create a 100x100 RGB image with some patterns
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some colored regions
        image[0:50, 0:50] = [255, 0, 0]  # Blue square
        image[50:100, 50:100] = [0, 255, 0]  # Green square
        image[0:50, 50:100] = [0, 0, 255]  # Red square
        return image
    
    @pytest.fixture
    def sample_grayscale_image(self):
        """Create a sample grayscale image."""
        # Create a 100x100 grayscale image with gradient
        image = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            image[i, :] = i * 2  # Gradient from 0 to ~200
        return image
    
    def test_validate_format_png(self, processor):
        """Test that PNG format is validated as supported."""
        path = Path("test_image.png")
        assert processor.validate_format(path) is True
    
    def test_validate_format_jpg(self, processor):
        """Test that JPG format is validated as supported."""
        path = Path("test_image.jpg")
        assert processor.validate_format(path) is True
    
    def test_validate_format_jpeg(self, processor):
        """Test that JPEG format is validated as supported."""
        path = Path("test_image.jpeg")
        assert processor.validate_format(path) is True
    
    def test_validate_format_bmp(self, processor):
        """Test that BMP format is validated as supported."""
        path = Path("test_image.bmp")
        assert processor.validate_format(path) is True
    
    def test_validate_format_unsupported(self, processor):
        """Test that unsupported formats are rejected."""
        unsupported_formats = [".gif", ".tiff", ".webp", ".svg", ".txt"]
        for fmt in unsupported_formats:
            path = Path(f"test_image{fmt}")
            assert processor.validate_format(path) is False
    
    def test_validate_format_case_insensitive(self, processor):
        """Test that format validation is case-insensitive."""
        paths = [Path("test.PNG"), Path("test.JPG"), Path("test.BMP")]
        for path in paths:
            assert processor.validate_format(path) is True
    
    def test_load_image_file_not_found(self, processor):
        """Test that loading nonexistent file raises ImageProcessingError."""
        path = Path("nonexistent_image.png")
        with pytest.raises(ImageProcessingError) as exc_info:
            processor.load_image(path)
        assert "not found" in str(exc_info.value)
    
    def test_load_image_unsupported_format(self, processor, sample_color_image):
        """Test that loading unsupported format raises ImageProcessingError."""
        with NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ImageProcessingError) as exc_info:
                processor.load_image(temp_path)
            assert "Unsupported image format" in str(exc_info.value)
        finally:
            temp_path.unlink()
    
    def test_load_image_png(self, processor, sample_color_image):
        """Test loading PNG image format."""
        with NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save sample image
            cv2.imwrite(str(temp_path), sample_color_image)
            
            # Load image
            loaded = processor.load_image(temp_path)
            
            assert loaded is not None
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape == sample_color_image.shape
            assert loaded.dtype == np.uint8
        finally:
            temp_path.unlink()
    
    def test_load_image_jpg(self, processor, sample_color_image):
        """Test loading JPG image format."""
        with NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save sample image
            cv2.imwrite(str(temp_path), sample_color_image)
            
            # Load image
            loaded = processor.load_image(temp_path)
            
            assert loaded is not None
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape == sample_color_image.shape
            assert loaded.dtype == np.uint8
        finally:
            temp_path.unlink()
    
    def test_load_image_bmp(self, processor, sample_color_image):
        """Test loading BMP image format."""
        with NamedTemporaryFile(suffix='.bmp', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save sample image
            cv2.imwrite(str(temp_path), sample_color_image)
            
            # Load image
            loaded = processor.load_image(temp_path)
            
            assert loaded is not None
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape == sample_color_image.shape
            assert loaded.dtype == np.uint8
        finally:
            temp_path.unlink()
    
    def test_preprocess_grayscale_conversion_color_image(self, processor, sample_color_image):
        """Test grayscale conversion from color image."""
        config = PreprocessConfig(
            target_size=(100, 100),
            normalize=False,
            denoise=False
        )
        
        result = processor.preprocess(sample_color_image, config)
        
        # Should be grayscale (2D array)
        assert result.ndim == 2
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
    
    def test_preprocess_grayscale_conversion_already_gray(self, processor, sample_grayscale_image):
        """Test grayscale conversion when image is already grayscale."""
        config = PreprocessConfig(
            target_size=(100, 100),
            normalize=False,
            denoise=False
        )
        
        result = processor.preprocess(sample_grayscale_image, config)
        
        # Should remain grayscale
        assert result.ndim == 2
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
    
    def test_preprocess_resize_without_aspect_ratio(self, processor, sample_color_image):
        """Test resizing without maintaining aspect ratio."""
        config = PreprocessConfig(
            target_size=(50, 75),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False
        )
        
        result = processor.preprocess(sample_color_image, config)
        
        # Should be resized to exact target size
        assert result.shape == (75, 50)  # (height, width)
    
    def test_preprocess_resize_with_aspect_ratio(self, processor, sample_color_image):
        """Test resizing with aspect ratio preservation."""
        config = PreprocessConfig(
            target_size=(50, 50),
            maintain_aspect_ratio=True,
            normalize=False,
            denoise=False
        )
        
        result = processor.preprocess(sample_color_image, config)
        
        # Should be padded to target size
        assert result.shape == (50, 50)
    
    def test_preprocess_normalization(self, processor, sample_color_image):
        """Test image normalization to [0, 1] range."""
        config = PreprocessConfig(
            target_size=(100, 100),
            normalize=True,
            denoise=False
        )
        
        result = processor.preprocess(sample_color_image, config)
        
        # Should be normalized to float32 in [0, 1]
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_preprocess_denoising(self, processor, sample_color_image):
        """Test denoising operation."""
        config = PreprocessConfig(
            target_size=(100, 100),
            normalize=False,
            denoise=True,
            denoise_strength=0.5
        )
        
        result = processor.preprocess(sample_color_image, config)
        
        # Should complete without error
        assert result is not None
        assert result.shape == (100, 100)
    
    def test_preprocess_all_operations(self, processor, sample_color_image):
        """Test all preprocessing operations together."""
        config = PreprocessConfig(
            target_size=(80, 80),
            maintain_aspect_ratio=True,
            normalize=True,
            denoise=True,
            denoise_strength=0.3
        )
        
        result = processor.preprocess(sample_color_image, config)
        
        # Should be grayscale, resized, denoised, and normalized
        assert result.ndim == 2
        assert result.shape == (80, 80)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_preprocess_single_pixel_image(self, processor):
        """Test preprocessing a single-pixel image."""
        # Create 1x1 image
        single_pixel = np.array([[[128, 128, 128]]], dtype=np.uint8)
        
        config = PreprocessConfig(
            target_size=(10, 10),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=False
        )
        
        result = processor.preprocess(single_pixel, config)
        
        # Should be resized to target size
        assert result.shape == (10, 10)
    
    def test_preprocess_empty_image(self, processor):
        """Test preprocessing a completely black (empty) image."""
        # Create all-black image
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        config = PreprocessConfig(
            target_size=(50, 50),
            normalize=False,
            denoise=False
        )
        
        result = processor.preprocess(empty_image, config)
        
        # Should complete without error
        assert result is not None
        assert result.shape == (50, 50)
        # All pixels should be zero
        assert np.all(result == 0)
    
    def test_preprocess_uniform_white_image(self, processor):
        """Test preprocessing a completely white (uniform) image."""
        # Create all-white image
        white_image = np.full((100, 100, 3), 255, dtype=np.uint8)
        
        config = PreprocessConfig(
            target_size=(50, 50),
            normalize=True,
            denoise=False
        )
        
        result = processor.preprocess(white_image, config)
        
        # Should complete without error
        assert result is not None
        assert result.shape == (50, 50)
        # All pixels should be close to 1.0 (normalized white)
        assert np.allclose(result, 1.0, atol=0.01)


class TestContourExtractor:
    """Tests for ContourExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a ContourExtractor instance."""
        return ContourExtractor()
    
    @pytest.fixture
    def simple_edge_map(self):
        """Create a simple edge map with a square."""
        edge_map = np.zeros((100, 100), dtype=np.uint8)
        # Draw a square
        edge_map[20:80, 20] = 255  # Left edge
        edge_map[20:80, 80] = 255  # Right edge
        edge_map[20, 20:80] = 255  # Top edge
        edge_map[80, 20:80] = 255  # Bottom edge
        return edge_map
    
    def test_extract_contours_simple(self, extractor, simple_edge_map):
        """Test extracting contours from simple edge map."""
        contours = extractor.extract_contours(simple_edge_map)
        
        assert len(contours) > 0
        assert all(isinstance(c, Contour) for c in contours)
        assert all(c.length >= 10 for c in contours)
    
    def test_extract_contours_empty_edge_map(self, extractor):
        """Test that empty edge map raises ImageProcessingError."""
        empty_map = np.zeros((100, 100), dtype=np.uint8)
        
        with pytest.raises(ImageProcessingError) as exc_info:
            extractor.extract_contours(empty_map)
        assert "empty" in str(exc_info.value).lower()
    
    def test_extract_contours_sorted_by_length(self, extractor, simple_edge_map):
        """Test that contours are sorted by length (descending)."""
        contours = extractor.extract_contours(simple_edge_map)
        
        # Check that contours are sorted by length
        lengths = [c.length for c in contours]
        assert lengths == sorted(lengths, reverse=True)
    
    def test_to_complex_plane(self, extractor):
        """Test conversion of contour points to complex plane."""
        # Create a simple contour
        points = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        contour = Contour(points=points, is_closed=False, length=3)
        
        complex_points = extractor.to_complex_plane(contour)
        
        assert len(complex_points) == 3
        assert complex_points.dtype == np.complex128
        # Check first point: 10 + 20i
        assert complex_points[0] == 10.0 + 20.0j
        # Check second point: 30 + 40i
        assert complex_points[1] == 30.0 + 40.0j
        # Check third point: 50 + 60i
        assert complex_points[2] == 50.0 + 60.0j
    
    def test_to_complex_plane_round_trip(self, extractor):
        """Test that converting to complex and back preserves coordinates."""
        # Create a contour with known points
        points = np.array([[1.5, 2.5], [3.7, 4.9], [5.1, 6.3]])
        contour = Contour(points=points, is_closed=False, length=3)
        
        # Convert to complex
        complex_points = extractor.to_complex_plane(contour)
        
        # Convert back to (x, y)
        x_recovered = complex_points.real
        y_recovered = complex_points.imag
        points_recovered = np.column_stack([x_recovered, y_recovered])
        
        # Should match original points
        assert np.allclose(points_recovered, points)
    
    def test_resample_contour_same_length(self, extractor):
        """Test resampling contour to same length returns copy."""
        # Create contour with at least 10 points (minimum required)
        t = np.linspace(0, 2*np.pi, 20, endpoint=False)
        points = np.column_stack([np.cos(t), np.sin(t)])
        contour = Contour(points=points, is_closed=True, length=20)
        
        resampled = extractor.resample_contour(contour, num_points=20)
        
        assert resampled.length == 20
        assert np.allclose(resampled.points, points)
    
    def test_resample_contour_increase_points(self, extractor):
        """Test resampling contour to more points."""
        points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        contour = Contour(points=points, is_closed=True, length=4)
        
        resampled = extractor.resample_contour(contour, num_points=20)
        
        assert resampled.length == 20
        assert resampled.is_closed == contour.is_closed
    
    def test_resample_contour_decrease_points(self, extractor):
        """Test resampling contour to fewer points."""
        # Create contour with many points
        t = np.linspace(0, 2*np.pi, 100, endpoint=False)
        points = np.column_stack([np.cos(t), np.sin(t)])
        contour = Contour(points=points, is_closed=True, length=100)
        
        resampled = extractor.resample_contour(contour, num_points=20)
        
        assert resampled.length == 20
        assert resampled.is_closed == contour.is_closed
    
    def test_resample_contour_minimum_points(self, extractor):
        """Test that resampling with less than 10 points raises error."""
        points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        contour = Contour(points=points, is_closed=False, length=3)
        
        with pytest.raises(ImageProcessingError) as exc_info:
            extractor.resample_contour(contour, num_points=5)
        assert "at least 10" in str(exc_info.value)
    
    def test_resample_contour_maximum_points(self, extractor):
        """Test that resampling with more than 10000 points raises error."""
        points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        contour = Contour(points=points, is_closed=False, length=3)
        
        with pytest.raises(ImageProcessingError) as exc_info:
            extractor.resample_contour(contour, num_points=15000)
        assert "maximum" in str(exc_info.value).lower()
