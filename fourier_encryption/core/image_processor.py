"""
Image processing pipeline for Fourier-Based Image Encryption System.

This module provides abstract base classes and concrete implementations for
image loading, preprocessing, and validation operations.

Performance optimizations:
- Cached format validation
- Optimized resize operations with appropriate interpolation
- Vectorized normalization
- Efficient memory usage with in-place operations where possible
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np

from fourier_encryption.config.settings import PreprocessConfig
from fourier_encryption.models.exceptions import ImageProcessingError


class ImageProcessor(ABC):
    """
    Abstract base class for image preprocessing operations.
    
    Defines the interface for loading, preprocessing, and validating images
    before they are processed by the Fourier transform pipeline.
    """
    
    @abstractmethod
    def load_image(self, path: Path) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            path: Path to image file
            
        Returns:
            NumPy array containing image data
            
        Raises:
            ImageProcessingError: If image cannot be loaded
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
        """
        Preprocess image according to configuration.
        
        Operations may include:
        - Grayscale conversion
        - Resizing (with optional aspect ratio preservation)
        - Normalization
        - Denoising
        
        Args:
            image: Input image as NumPy array
            config: Preprocessing configuration
            
        Returns:
            Preprocessed image as NumPy array
            
        Raises:
            ImageProcessingError: If preprocessing fails
        """
        pass
    
    @abstractmethod
    def validate_format(self, path: Path) -> bool:
        """
        Validate that the image file format is supported.
        
        Args:
            path: Path to image file
            
        Returns:
            True if format is supported, False otherwise
        """
        pass


class OpenCVImageProcessor(ImageProcessor):
    """
    Concrete implementation of ImageProcessor using OpenCV.
    
    Supports PNG, JPG, and BMP formats with configurable preprocessing
    including grayscale conversion, resizing, normalization, and denoising.
    Handles images up to 4K resolution.
    
    Performance optimizations:
    - Cached format validation
    - Optimized interpolation method selection
    - Vectorized normalization
    - Efficient memory usage
    """
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp'}
    MAX_DIMENSION = 3840  # 4K resolution width
    
    def __init__(self):
        """Initialize processor with caching support."""
        self._format_cache = {}
    
    def validate_format(self, path: Path) -> bool:
        """
        Validate that the image file format is supported.
        
        Uses caching to avoid repeated suffix checks.
        
        Args:
            path: Path to image file
            
        Returns:
            True if format is supported (PNG, JPG, BMP), False otherwise
        """
        suffix = path.suffix.lower()
        if suffix not in self._format_cache:
            self._format_cache[suffix] = suffix in self.SUPPORTED_FORMATS
        return self._format_cache[suffix]
    
    def load_image(self, path: Path) -> np.ndarray:
        """
        Load image from file using OpenCV.
        
        Args:
            path: Path to image file
            
        Returns:
            NumPy array containing image data in BGR format
            
        Raises:
            ImageProcessingError: If file not found, format unsupported, or loading fails
        """
        if not path.exists():
            raise ImageProcessingError(
                f"Image file not found: {path}",
                context={"path": str(path)}
            )
        
        if not self.validate_format(path):
            raise ImageProcessingError(
                f"Unsupported image format: {path.suffix}",
                context={
                    "path": str(path),
                    "format": path.suffix,
                    "supported": list(self.SUPPORTED_FORMATS)
                }
            )
        
        try:
            image = cv2.imread(str(path))
            
            if image is None:
                raise ImageProcessingError(
                    f"Failed to load image (corrupted or invalid file): {path}",
                    context={"path": str(path)}
                )
            
            # Validate image dimensions
            height, width = image.shape[:2]
            if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
                raise ImageProcessingError(
                    f"Image dimensions exceed maximum supported size",
                    context={
                        "path": str(path),
                        "dimensions": f"{width}x{height}",
                        "max_dimension": self.MAX_DIMENSION
                    }
                )
            
            return image
            
        except cv2.error as e:
            raise ImageProcessingError(
                f"OpenCV error while loading image: {e}",
                context={"path": str(path), "error": str(e)}
            )
    
    def preprocess(self, image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
        """
        Preprocess image according to configuration.
        
        Applies the following operations in order:
        1. Grayscale conversion
        2. Resizing (with optional aspect ratio preservation)
        3. Denoising (if enabled)
        4. Normalization (if enabled)
        
        Optimizations:
        - Efficient interpolation method selection based on scaling direction
        - Vectorized normalization
        - Pre-allocated arrays for padding
        
        Args:
            image: Input image as NumPy array (BGR format from OpenCV)
            config: Preprocessing configuration
            
        Returns:
            Preprocessed grayscale image as NumPy array
            
        Raises:
            ImageProcessingError: If preprocessing operations fail
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize if needed
            current_height, current_width = gray.shape
            target_width, target_height = config.target_size
            
            if (current_width, current_height) != (target_width, target_height):
                if config.maintain_aspect_ratio:
                    # Calculate scaling factor to fit within target size
                    scale = min(target_width / current_width, target_height / current_height)
                    new_width = int(current_width * scale)
                    new_height = int(current_height * scale)
                    
                    # Choose interpolation method based on scaling direction
                    # INTER_AREA is best for downscaling, INTER_LINEAR for upscaling
                    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    
                    # Resize with aspect ratio preserved
                    resized = cv2.resize(gray, (new_width, new_height), interpolation=interpolation)
                    
                    # Pad to target size if needed
                    if (new_width, new_height) != (target_width, target_height):
                        # Create black canvas of target size (pre-allocated)
                        padded = np.zeros((target_height, target_width), dtype=gray.dtype)
                        
                        # Calculate padding offsets to center the image
                        y_offset = (target_height - new_height) // 2
                        x_offset = (target_width - new_width) // 2
                        
                        # Place resized image in center
                        padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
                        gray = padded
                    else:
                        gray = resized
                else:
                    # Choose interpolation method based on scaling direction
                    scale = min(target_width / current_width, target_height / current_height)
                    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    
                    # Resize without preserving aspect ratio
                    gray = cv2.resize(gray, (target_width, target_height), interpolation=interpolation)
            
            # Apply denoising if enabled
            if config.denoise:
                # Use Non-local Means Denoising
                h = int(config.denoise_strength * 20)  # Scale strength to OpenCV parameter range
                gray = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)
            
            # Normalize if enabled (vectorized operation)
            if config.normalize:
                # Vectorized normalization to [0, 1] range
                # This is faster than element-wise division
                gray = np.multiply(gray, 1.0 / 255.0, dtype=np.float32)
            
            return gray
            
        except cv2.error as e:
            raise ImageProcessingError(
                f"OpenCV error during preprocessing: {e}",
                context={"error": str(e)}
            )
        except Exception as e:
            raise ImageProcessingError(
                f"Unexpected error during preprocessing: {e}",
                context={"error": str(e)}
            )
