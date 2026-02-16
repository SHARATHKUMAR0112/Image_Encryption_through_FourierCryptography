"""
Edge detection strategies for Fourier-Based Image Encryption System.

This module provides abstract base classes and concrete implementations for
edge detection operations, including traditional Canny edge detection and
AI-enhanced detection methods.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict

import cv2
import numpy as np

from fourier_encryption.models.exceptions import ImageProcessingError
from fourier_encryption.models.data_models import EdgeDetectionConfig


class EdgeDetector(ABC):
    """
    Abstract base class for edge detection strategies.
    
    Defines the interface for detecting edges in preprocessed images.
    Implementations should return binary edge maps and track performance metrics.
    """
    
    @abstractmethod
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the input image.
        
        Args:
            image: Input image as NumPy array (grayscale)
            
        Returns:
            Binary edge map as NumPy array (0 or 255 values)
            
        Raises:
            ImageProcessingError: If edge detection fails
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the last edge detection operation.
        
        Returns:
            Dictionary containing metrics such as:
                - detection_time: Time taken for edge detection in seconds
                - edge_density: Percentage of edge pixels in the output
                - image_size: Number of pixels processed
        """
        pass



class CannyEdgeDetector(EdgeDetector):
    """
    Canny edge detection with adaptive thresholding.
    
    Implements the Canny edge detection algorithm using OpenCV with automatic
    threshold calculation based on image statistics. Tracks performance metrics
    for monitoring and optimization.
    
    Attributes:
        _last_detection_time: Time taken for the last edge detection operation
        _last_edge_density: Percentage of edge pixels in the last output
        _last_image_size: Number of pixels in the last processed image
    """
    
    def __init__(self, low_threshold: float = 50, high_threshold: float = 150,
                 aperture_size: int = 3, use_adaptive_threshold: bool = True):
        """
        Initialize Canny edge detector.
        
        Args:
            low_threshold: Lower threshold for edge detection (default: 50)
            high_threshold: Upper threshold for edge detection (default: 150)
            aperture_size: Aperture size for Sobel operator (default: 3, must be odd)
            use_adaptive_threshold: If True, calculate thresholds from image statistics
        """
        if aperture_size % 2 == 0 or aperture_size < 3 or aperture_size > 7:
            raise ValueError(
                f"aperture_size must be odd and in range [3, 7], got {aperture_size}"
            )
        
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.use_adaptive_threshold = use_adaptive_threshold
        
        # Performance metrics
        self._last_detection_time = 0.0
        self._last_edge_density = 0.0
        self._last_image_size = 0
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny algorithm with optional adaptive thresholding.
        
        If adaptive thresholding is enabled, thresholds are calculated based on
        the median pixel intensity:
            - low_threshold = max(0, 0.66 * median)
            - high_threshold = min(255, 1.33 * median)
        
        Args:
            image: Input grayscale image as NumPy array
            
        Returns:
            Binary edge map as NumPy array (0 or 255 values)
            
        Raises:
            ImageProcessingError: If edge detection fails or input is invalid
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not isinstance(image, np.ndarray):
                raise ImageProcessingError(
                    "Input must be a NumPy array",
                    context={"type": type(image).__name__}
                )
            
            if image.size == 0:
                raise ImageProcessingError(
                    "Input image is empty",
                    context={"shape": image.shape}
                )
            
            # Convert to uint8 if needed (Canny requires uint8)
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume normalized [0, 1] range
                if image.max() <= 1.0:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = image.astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
            
            # Ensure grayscale (2D array)
            if image_uint8.ndim != 2:
                raise ImageProcessingError(
                    "Input image must be grayscale (2D array)",
                    context={"shape": image_uint8.shape, "ndim": image_uint8.ndim}
                )
            
            # Calculate adaptive thresholds if enabled
            if self.use_adaptive_threshold:
                median = np.median(image_uint8)
                low_thresh = max(0, int(0.66 * median))
                high_thresh = min(255, int(1.33 * median))
            else:
                low_thresh = int(self.low_threshold)
                high_thresh = int(self.high_threshold)
            
            # Apply Canny edge detection
            edges = cv2.Canny(
                image_uint8,
                threshold1=low_thresh,
                threshold2=high_thresh,
                apertureSize=self.aperture_size
            )
            
            # Update performance metrics
            self._last_detection_time = time.time() - start_time
            self._last_image_size = image_uint8.size
            self._last_edge_density = (np.count_nonzero(edges) / edges.size) * 100.0
            
            return edges
            
        except cv2.error as e:
            raise ImageProcessingError(
                f"OpenCV error during Canny edge detection: {e}",
                context={"error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(
                f"Unexpected error during edge detection: {e}",
                context={"error": str(e), "error_type": type(e).__name__}
            )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the last edge detection operation.
        
        Returns:
            Dictionary containing:
                - detection_time: Time taken in seconds
                - edge_density: Percentage of edge pixels (0-100)
                - image_size: Number of pixels processed
        """
        return {
            "detection_time": self._last_detection_time,
            "edge_density": self._last_edge_density,
            "image_size": float(self._last_image_size)
        }



class IndustrialEdgeDetector(EdgeDetector):
    """
    Production-grade edge detection with robust foreground extraction.
    
    This detector implements a comprehensive pipeline optimized for industrial use:
    1. GrabCut-based foreground extraction with mask refinement
    2. Adaptive Gaussian preprocessing with odd kernel enforcement
    3. Tunable Canny edge detection
    4. Morphological cleanup using elliptical kernel (closing operation)
    
    The pipeline is designed to produce sketch-ready output with clean, connected
    edges suitable for Fourier decomposition.
    
    Attributes:
        config: EdgeDetectionConfig with tunable parameters
        _last_detection_time: Time taken for the last edge detection operation
        _last_edge_density: Percentage of edge pixels in the last output
        _last_image_size: Number of pixels in the last processed image
    """
    
    def __init__(self, config: EdgeDetectionConfig = None):
        """
        Initialize industrial edge detector.
        
        Args:
            config: EdgeDetectionConfig with pipeline parameters.
                   If None, uses default configuration.
        """
        self.config = config if config is not None else EdgeDetectionConfig()
        
        # Performance metrics
        self._last_detection_time = 0.0
        self._last_edge_density = 0.0
        self._last_image_size = 0
    
    def extract_foreground(self, image: np.ndarray) -> np.ndarray:
        """
        Perform foreground extraction using GrabCut algorithm.
        
        Uses rectangular initialization with configurable iterations.
        Refines mask to separate foreground (GC_FGD, GC_PR_FGD) from background.
        
        Args:
            image: Input image as NumPy array (BGR or grayscale)
            
        Returns:
            Foreground mask as binary NumPy array (0 or 255)
            
        Raises:
            ImageProcessingError: If foreground extraction fails
        """
        try:
            # Ensure image is in BGR format for GrabCut
            if image.ndim == 2:
                # Convert grayscale to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 3:
                image_bgr = image
            else:
                raise ImageProcessingError(
                    f"Invalid image shape for foreground extraction: {image.shape}",
                    context={"shape": image.shape}
                )
            
            # Initialize rectangle for GrabCut (leave small margin)
            height, width = image_bgr.shape[:2]
            margin = 10
            rect = (margin, margin, width - 2 * margin, height - 2 * margin)
            
            # Initialize mask and models for GrabCut
            mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)
            
            # Apply GrabCut algorithm
            cv2.grabCut(
                image_bgr,
                mask,
                rect,
                bgd_model,
                fgd_model,
                self.config.grabcut_iterations,
                cv2.GC_INIT_WITH_RECT
            )
            
            # Create binary mask: foreground (GC_FGD=1, GC_PR_FGD=3) vs background
            foreground_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                255,
                0
            ).astype(np.uint8)
            
            return foreground_mask
            
        except cv2.error as e:
            raise ImageProcessingError(
                f"OpenCV error during GrabCut foreground extraction: {e}",
                context={"error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(
                f"Unexpected error during foreground extraction: {e}",
                context={"error": str(e), "error_type": type(e).__name__}
            )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image: grayscale conversion + Gaussian blur.
        Ensures odd kernel size for Gaussian blur.
        
        Args:
            image: Input image as NumPy array (BGR or grayscale)
            
        Returns:
            Preprocessed grayscale image as NumPy array
            
        Raises:
            ImageProcessingError: If preprocessing fails
        """
        try:
            # Convert to grayscale if needed
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Ensure kernel size is odd (already validated in config)
            kernel_size = self.config.gaussian_kernel
            
            # Apply Gaussian blur for noise reduction
            blurred = cv2.GaussianBlur(
                gray,
                (kernel_size, kernel_size),
                0  # Let OpenCV calculate sigma automatically
            )
            
            return blurred
            
        except cv2.error as e:
            raise ImageProcessingError(
                f"OpenCV error during preprocessing: {e}",
                context={"error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(
                f"Unexpected error during preprocessing: {e}",
                context={"error": str(e), "error_type": type(e).__name__}
            )
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection with configurable thresholds.
        
        Args:
            image: Input grayscale image as NumPy array
            
        Returns:
            Binary edge map as NumPy array (0 or 255)
            
        Raises:
            ImageProcessingError: If edge detection fails
        """
        try:
            # Validate input
            if not isinstance(image, np.ndarray):
                raise ImageProcessingError(
                    "Input must be a NumPy array",
                    context={"type": type(image).__name__}
                )
            
            if image.size == 0:
                raise ImageProcessingError(
                    "Input image is empty",
                    context={"shape": image.shape}
                )
            
            # Ensure grayscale (2D array)
            if image.ndim != 2:
                raise ImageProcessingError(
                    "Input image must be grayscale (2D array)",
                    context={"shape": image.shape, "ndim": image.ndim}
                )
            
            # Convert to uint8 if needed
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = image.astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
            
            # Apply Canny edge detection with configured thresholds
            edges = cv2.Canny(
                image_uint8,
                threshold1=self.config.canny_threshold1,
                threshold2=self.config.canny_threshold2
            )
            
            return edges
            
        except cv2.error as e:
            raise ImageProcessingError(
                f"OpenCV error during Canny edge detection: {e}",
                context={"error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(
                f"Unexpected error during edge detection: {e}",
                context={"error": str(e), "error_type": type(e).__name__}
            )
    
    def postprocess(self, edges: np.ndarray) -> np.ndarray:
        """
        Apply morphological refinement using elliptical kernel.
        Performs closing operation to connect nearby edges.
        
        Args:
            edges: Binary edge map as NumPy array
            
        Returns:
            Refined edge map as NumPy array
            
        Raises:
            ImageProcessingError: If postprocessing fails
        """
        try:
            # Create elliptical structuring element
            kernel_size = self.config.morph_kernel_size
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size)
            )
            
            # Apply morphological closing (dilation followed by erosion)
            # This connects nearby edges and fills small gaps
            refined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            return refined
            
        except cv2.error as e:
            raise ImageProcessingError(
                f"OpenCV error during morphological postprocessing: {e}",
                context={"error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(
                f"Unexpected error during postprocessing: {e}",
                context={"error": str(e), "error_type": type(e).__name__}
            )
    
    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Full pipeline: foreground extraction → preprocess → 
        edge detection → postprocess.
        
        This is the main entry point for the industrial edge detection pipeline.
        Each stage can be enabled/disabled via the configuration.
        
        Args:
            image: Input image as NumPy array (BGR or grayscale)
            
        Returns:
            Binary edge map as NumPy array (0 or 255)
            
        Raises:
            ImageProcessingError: If any pipeline stage fails
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not isinstance(image, np.ndarray):
                raise ImageProcessingError(
                    "Input must be a NumPy array",
                    context={"type": type(image).__name__}
                )
            
            if image.size == 0:
                raise ImageProcessingError(
                    "Input image is empty",
                    context={"shape": image.shape}
                )
            
            # Stage 1: Foreground extraction (optional)
            if self.config.enable_foreground_extraction:
                foreground_mask = self.extract_foreground(image)
                # Apply mask to image
                if image.ndim == 3:
                    masked_image = cv2.bitwise_and(image, image, mask=foreground_mask)
                else:
                    masked_image = cv2.bitwise_and(image, image, mask=foreground_mask)
            else:
                masked_image = image
            
            # Stage 2: Preprocessing (grayscale + Gaussian blur)
            preprocessed = self.preprocess(masked_image)
            
            # Stage 3: Edge detection (Canny)
            edges = self.detect_edges(preprocessed)
            
            # Stage 4: Morphological refinement (optional)
            if self.config.enable_morphology:
                final_edges = self.postprocess(edges)
            else:
                final_edges = edges
            
            # Update performance metrics
            self._last_detection_time = time.time() - start_time
            self._last_image_size = image.size
            self._last_edge_density = (np.count_nonzero(final_edges) / final_edges.size) * 100.0
            
            return final_edges
            
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError(
                f"Unexpected error in industrial edge detection pipeline: {e}",
                context={"error": str(e), "error_type": type(e).__name__}
            )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the last edge detection operation.
        
        Returns:
            Dictionary containing:
                - detection_time: Time taken in seconds
                - edge_density: Percentage of edge pixels (0-100)
                - image_size: Number of pixels processed
        """
        return {
            "detection_time": self._last_detection_time,
            "edge_density": self._last_edge_density,
            "image_size": float(self._last_image_size)
        }
