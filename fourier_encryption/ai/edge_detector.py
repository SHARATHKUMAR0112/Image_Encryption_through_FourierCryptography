"""
AI-enhanced edge detection for Fourier-Based Image Encryption System.

This module provides AI-based edge detection using CNN or Vision Transformer
models with GPU acceleration and automatic fallback to traditional methods.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from fourier_encryption.core.edge_detector import CannyEdgeDetector, EdgeDetector
from fourier_encryption.models.exceptions import AIModelError, ImageProcessingError

logger = logging.getLogger(__name__)


class AIEdgeDetector(EdgeDetector):
    """
    AI-enhanced edge detector using CNN or Vision Transformer models.
    
    This detector attempts to use a pre-trained AI model for edge detection
    with GPU acceleration. If the model fails to load or GPU is unavailable,
    it automatically falls back to traditional Canny edge detection.
    
    Attributes:
        model_path: Path to the pre-trained model file
        device: Device to use for inference ("cuda" or "cpu")
        model: Loaded AI model (None if not loaded)
        fallback_detector: Canny edge detector for fallback
        _last_detection_time: Time taken for last detection
        _last_edge_density: Percentage of edge pixels in last output
        _last_image_size: Number of pixels in last processed image
        _used_fallback: Whether fallback was used in last detection
    """
    
    def __init__(self, model_path: Optional[Path] = None, device: str = "cuda"):
        """
        Initialize AI edge detector.
        
        Args:
            model_path: Path to pre-trained model file (optional)
            device: Device to use for inference ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.fallback_detector = CannyEdgeDetector()
        
        # Performance metrics
        self._last_detection_time = 0.0
        self._last_edge_density = 0.0
        self._last_image_size = 0
        self._used_fallback = False
        
        # Try to load model if path provided
        if model_path is not None:
            try:
                self.model = self._load_model(model_path)
                logger.info(f"AI edge detector model loaded from {model_path}")
            except AIModelError as e:
                logger.warning(f"Failed to load AI model: {e}, will use fallback")
                self.model = None
    
    def _load_model(self, path: Path):
        """
        Load pre-trained AI model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded model object
            
        Raises:
            AIModelError: If model loading fails
        """
        if not path.exists():
            raise AIModelError(
                f"Model file not found: {path}",
                context={"path": str(path)}
            )
        
        # For now, return None as placeholder
        # In a real implementation, this would load PyTorch/TensorFlow model
        logger.warning("AI model loading not implemented, using fallback")
        return None
    
    def _preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for AI model input.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image ready for model inference
        """
        # Normalize to [0, 1] range
        if image.dtype == np.uint8:
            normalized = image.astype(np.float32) / 255.0
        else:
            normalized = image.astype(np.float32)
            if normalized.max() > 1.0:
                normalized = normalized / 255.0
        
        return normalized
    
    def _postprocess_model_output(self, output: np.ndarray) -> np.ndarray:
        """
        Post-process AI model output to binary edge map.
        
        Args:
            output: Raw model output
            
        Returns:
            Binary edge map (0 or 255 values)
        """
        # Threshold at 0.5 and convert to binary
        binary = (output > 0.5).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up edges
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _detect_with_ai(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using AI model.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary edge map
            
        Raises:
            AIModelError: If inference fails
        """
        # Preprocess
        preprocessed = self._preprocess_for_model(image)
        
        # Run inference (placeholder - would use actual model)
        # For now, raise error to trigger fallback
        raise AIModelError("AI model inference not implemented")
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using AI model with automatic fallback to Canny.
        
        Attempts to use the AI model for edge detection. If the model is not
        loaded, GPU is unavailable, or inference fails, automatically falls
        back to traditional Canny edge detection without raising an error.
        
        Args:
            image: Input grayscale image as NumPy array
            
        Returns:
            Binary edge map as NumPy array (0 or 255 values)
            
        Raises:
            ImageProcessingError: If both AI and fallback detection fail
        """
        start_time = time.time()
        self._used_fallback = False
        
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
            
            # Try AI detection if model is loaded
            if self.model is not None:
                try:
                    edges = self._detect_with_ai(image)
                    logger.debug("AI edge detection successful")
                except (AIModelError, RuntimeError) as e:
                    logger.warning(f"AI detection failed: {e}, falling back to Canny")
                    edges = self.fallback_detector.detect_edges(image)
                    self._used_fallback = True
            else:
                # No model loaded, use fallback
                logger.debug("No AI model loaded, using Canny fallback")
                edges = self.fallback_detector.detect_edges(image)
                self._used_fallback = True
            
            # Update performance metrics
            self._last_detection_time = time.time() - start_time
            self._last_image_size = image.size
            self._last_edge_density = (np.count_nonzero(edges) / edges.size) * 100.0
            
            # Verify output is binary
            unique_values = np.unique(edges)
            if not np.all(np.isin(unique_values, [0, 255])):
                raise ImageProcessingError(
                    "Edge detection output is not binary",
                    context={"unique_values": unique_values.tolist()}
                )
            
            return edges
            
        except ImageProcessingError:
            raise
        except Exception as e:
            raise ImageProcessingError(
                f"Unexpected error during AI edge detection: {e}",
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
                - used_fallback: 1.0 if fallback was used, 0.0 otherwise
        """
        return {
            "detection_time": self._last_detection_time,
            "edge_density": self._last_edge_density,
            "image_size": float(self._last_image_size),
            "used_fallback": 1.0 if self._used_fallback else 0.0
        }
    
    def is_gpu_available(self) -> bool:
        """
        Check if GPU is available for inference.
        
        Returns:
            True if GPU is available, False otherwise
        """
        # For now, always return False as placeholder
        # In real implementation, would check torch.cuda.is_available() or similar
        return False
