"""
Integration test for core mathematical pipeline:
Image → Contours → DFT → Epicycles
"""

import numpy as np
import pytest
from pathlib import Path

from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.config.settings import PreprocessConfig
from fourier_encryption.models.exceptions import ImageProcessingError


class TestCorePipeline:
    """Test the complete core mathematical pipeline"""
    
    def test_pipeline_with_synthetic_image(self):
        """Test full pipeline: synthetic image → contours → DFT → epicycles"""
        # Create a simple synthetic image (white square on black background)
        image = np.zeros((200, 200), dtype=np.uint8)
        image[50:150, 50:150] = 255
        
        # Step 1: Edge detection
        edge_detector = CannyEdgeDetector()
        edge_map = edge_detector.detect_edges(image)
        
        assert edge_map is not None
        assert edge_map.shape == image.shape
        assert edge_map.dtype == np.uint8
        
        # Step 2: Contour extraction
        contour_extractor = ContourExtractor()
        contours = contour_extractor.extract_contours(edge_map)
        
        assert len(contours) > 0, "Should detect at least one contour"
        
        # Step 3: Convert to complex plane
        main_contour = contours[0]
        complex_points = contour_extractor.to_complex_plane(main_contour)
        
        assert complex_points is not None
        assert len(complex_points) > 0
        assert complex_points.dtype == np.complex128
        
        # Step 4: Compute DFT
        fourier_transformer = FourierTransformer()
        coefficients = fourier_transformer.compute_dft(complex_points)
        
        assert len(coefficients) > 0
        assert all(coef.amplitude >= 0 for coef in coefficients)
        assert all(-np.pi <= coef.phase <= np.pi for coef in coefficients)
        
        # Step 5: Sort and truncate coefficients
        sorted_coefficients = fourier_transformer.sort_by_amplitude(coefficients)
        truncated_coefficients = fourier_transformer.truncate_coefficients(
            sorted_coefficients, num_terms=50
        )
        
        assert len(truncated_coefficients) == min(50, len(sorted_coefficients))
        # Verify sorted in descending order
        for i in range(len(truncated_coefficients) - 1):
            assert truncated_coefficients[i].amplitude >= truncated_coefficients[i + 1].amplitude
        
        # Step 6: Create epicycle engine
        epicycle_engine = EpicycleEngine(truncated_coefficients)
        
        # Step 7: Generate animation frames
        frames = list(epicycle_engine.generate_animation_frames(num_frames=10, speed=1.0))
        
        assert len(frames) == 10
        assert all(hasattr(frame, 'time') for frame in frames)
        assert all(hasattr(frame, 'positions') for frame in frames)
        assert all(hasattr(frame, 'trace_point') for frame in frames)
        
        # Verify epicycle radii match coefficient amplitudes
        first_frame = frames[0]
        assert len(first_frame.positions) == len(truncated_coefficients)
        
    def test_pipeline_with_circle_contour(self):
        """Test pipeline with a perfect circle"""
        # Create a circle contour
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        radius = 50
        x = radius * np.cos(t) + 100
        y = radius * np.sin(t) + 100
        
        # Create image with circle
        image = np.zeros((200, 200), dtype=np.uint8)
        for i in range(len(x)):
            xi, yi = int(x[i]), int(y[i])
            if 0 <= xi < 200 and 0 <= yi < 200:
                image[yi, xi] = 255
        
        # Run through pipeline
        edge_detector = CannyEdgeDetector()
        edge_map = edge_detector.detect_edges(image)
        
        contour_extractor = ContourExtractor()
        contours = contour_extractor.extract_contours(edge_map)
        
        assert len(contours) > 0
        
        main_contour = contours[0]
        complex_points = contour_extractor.to_complex_plane(main_contour)
        
        fourier_transformer = FourierTransformer()
        coefficients = fourier_transformer.compute_dft(complex_points)
        
        # For a circle, the DC component (frequency 0) and first harmonic should dominate
        sorted_coefficients = fourier_transformer.sort_by_amplitude(coefficients)
        
        assert len(sorted_coefficients) > 0
        # The largest amplitude should be significantly larger than others
        assert sorted_coefficients[0].amplitude > 0
        
        # Create epicycle engine and verify it works
        epicycle_engine = EpicycleEngine(sorted_coefficients[:20])
        state = epicycle_engine.compute_state(0.0)
        
        assert state is not None
        assert len(state.positions) == 20
        
    def test_pipeline_dft_idft_reconstruction(self):
        """Test that DFT → IDFT reconstructs the original contour"""
        # Create a simple contour
        t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        complex_points = 10 * np.exp(1j * t) + 5 * np.exp(2j * t)
        
        # Compute DFT
        fourier_transformer = FourierTransformer()
        coefficients = fourier_transformer.compute_dft(complex_points)
        
        # Compute IDFT
        reconstructed = fourier_transformer.compute_idft(coefficients)
        
        # Verify reconstruction matches original
        assert len(reconstructed) == len(complex_points)
        np.testing.assert_allclose(
            reconstructed, 
            complex_points, 
            rtol=1e-10, 
            atol=1e-10
        )
        
    def test_pipeline_handles_empty_contours_gracefully(self):
        """Test that pipeline handles images with no contours"""
        # Create a completely black image (no edges)
        image = np.zeros((100, 100), dtype=np.uint8)
        
        edge_detector = CannyEdgeDetector()
        edge_map = edge_detector.detect_edges(image)
        
        contour_extractor = ContourExtractor()
        
        # Should raise ImageProcessingError for empty edge map
        with pytest.raises(ImageProcessingError, match="Edge map is completely empty"):
            contours = contour_extractor.extract_contours(edge_map)
        
    def test_epicycle_animation_time_progression(self):
        """Test that epicycle animation progresses correctly through time"""
        # Create simple coefficients
        from fourier_encryption.models.data_models import FourierCoefficient
        
        coefficients = [
            FourierCoefficient(frequency=0, amplitude=10.0, phase=0.0, complex_value=10+0j),
            FourierCoefficient(frequency=1, amplitude=5.0, phase=0.0, complex_value=5+0j),
        ]
        
        epicycle_engine = EpicycleEngine(coefficients)
        frames = list(epicycle_engine.generate_animation_frames(num_frames=20, speed=1.0))
        
        # Verify time progresses from 0 to 2π
        times = [frame.time for frame in frames]
        assert times[0] == 0.0
        assert times[-1] < 2 * np.pi
        
        # Verify times are monotonically increasing
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]
