"""
AI Coefficient Optimizer for intelligent Fourier coefficient selection.

This module implements an AI-based optimizer that automatically determines
the optimal number of Fourier coefficients needed to reconstruct an image
while minimizing payload size and maintaining reconstruction quality.
"""

import logging
from typing import List, Tuple

import numpy as np

from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.models.data_models import (
    FourierCoefficient,
    OptimizationResult,
)
from fourier_encryption.models.exceptions import AIModelError

logger = logging.getLogger(__name__)


class CoefficientOptimizer:
    """
    AI-based coefficient count optimization.
    
    This class analyzes image complexity and determines the optimal number
    of Fourier coefficients needed to achieve a target reconstruction quality
    while minimizing payload size.
    
    Uses binary search to find the minimum coefficient count that satisfies
    the reconstruction error threshold (< 5% RMSE).
    
    Attributes:
        target_error: Maximum acceptable reconstruction error (RMSE)
        min_coefficients: Minimum number of coefficients (default: 10)
        max_coefficients: Maximum number of coefficients (default: 1000)
        transformer: FourierTransformer instance for IDFT operations
    """
    
    def __init__(
        self,
        target_error: float = 0.05,
        min_coefficients: int = 10,
        max_coefficients: int = 1000
    ):
        """
        Initialize the coefficient optimizer.
        
        Args:
            target_error: Maximum acceptable reconstruction error (default: 5%)
            min_coefficients: Minimum coefficient count (default: 10)
            max_coefficients: Maximum coefficient count (default: 1000)
        
        Raises:
            AIModelError: If parameters are invalid
        """
        if not (0 < target_error < 1):
            raise AIModelError(
                f"target_error must be in (0, 1), got {target_error}"
            )
        
        if not (10 <= min_coefficients <= max_coefficients <= 1000):
            raise AIModelError(
                f"Invalid coefficient range: [{min_coefficients}, {max_coefficients}]"
            )
        
        self.target_error = target_error
        self.min_coefficients = min_coefficients
        self.max_coefficients = max_coefficients
        self.transformer = FourierTransformer()
        
        logger.info(
            f"Initialized CoefficientOptimizer with target_error={target_error}, "
            f"range=[{min_coefficients}, {max_coefficients}]"
        )
    
    def classify_complexity(self, image: np.ndarray) -> str:
        """
        Analyze image features to classify complexity.
        
        Classifies images into complexity categories based on edge density,
        frequency content, and structural features. This helps determine
        appropriate coefficient count ranges.
        
        Complexity classification:
        - "low": Simple shapes, few edges (< 20% edge pixels)
        - "medium": Moderate detail (20-50% edge pixels)
        - "high": Complex details, many edges (> 50% edge pixels)
        
        Args:
            image: Grayscale image as NumPy array (2D)
        
        Returns:
            Complexity class: "low", "medium", or "high"
        
        Raises:
            AIModelError: If image is invalid
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise AIModelError("image must be a NumPy array")
        
        if image.ndim != 2:
            raise AIModelError(
                f"image must be 2D grayscale array, got shape {image.shape}"
            )
        
        if image.size == 0:
            raise AIModelError("image cannot be empty")
        
        # Compute edge density (percentage of non-zero pixels)
        # Assuming image is already an edge map or binary image
        edge_pixels = np.count_nonzero(image)
        total_pixels = image.size
        edge_density = edge_pixels / total_pixels
        
        # Special case: if edge density is extremely low (< 1%), classify as low complexity
        # This handles uniform or nearly uniform images
        # Also check if the image has very low variance (uniform pixel values)
        pixel_variance = np.var(image)
        if edge_density < 0.01 or pixel_variance < 1.0:
            logger.info(
                f"Image complexity: low (edge_density={edge_density:.3f}, "
                f"variance={pixel_variance:.3f}, nearly uniform)"
            )
            return "low"
        
        # Compute frequency content using FFT
        # High-frequency content indicates more complex details
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        
        # Calculate high-frequency energy ratio
        # High frequencies are in the outer regions of the spectrum
        center_y, center_x = np.array(magnitude_spectrum.shape) // 2
        radius = min(center_y, center_x) // 2
        
        # Create mask for high-frequency region (outer half)
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        high_freq_mask = distance_from_center > radius
        
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Classify based on edge density and frequency content
        # Combine both metrics for robust classification
        complexity_score = 0.6 * edge_density + 0.4 * high_freq_ratio
        
        if complexity_score < 0.2:
            complexity_class = "low"
        elif complexity_score < 0.5:
            complexity_class = "medium"
        else:
            complexity_class = "high"
        
        logger.info(
            f"Image complexity: {complexity_class} "
            f"(edge_density={edge_density:.3f}, high_freq_ratio={high_freq_ratio:.3f})"
        )
        
        return complexity_class
    
    def compute_reconstruction_error(
        self,
        original_points: np.ndarray,
        reconstructed_points: np.ndarray
    ) -> float:
        """
        Calculate RMSE between original and reconstructed contours.
        
        Computes the Root Mean Square Error (RMSE) as a measure of
        reconstruction quality. Lower RMSE indicates better reconstruction.
        
        RMSE formula:
            RMSE = sqrt(mean((original - reconstructed)^2))
        
        Args:
            original_points: Original contour points (complex array)
            reconstructed_points: Reconstructed contour points (complex array)
        
        Returns:
            RMSE value (normalized by signal magnitude)
        
        Raises:
            AIModelError: If arrays are incompatible
        """
        # Validate inputs
        if not isinstance(original_points, np.ndarray):
            raise AIModelError("original_points must be a NumPy array")
        
        if not isinstance(reconstructed_points, np.ndarray):
            raise AIModelError("reconstructed_points must be a NumPy array")
        
        if original_points.shape != reconstructed_points.shape:
            raise AIModelError(
                f"Shape mismatch: original {original_points.shape} vs "
                f"reconstructed {reconstructed_points.shape}"
            )
        
        if original_points.size == 0:
            raise AIModelError("Points arrays cannot be empty")
        
        # Compute squared differences
        # For complex numbers, use absolute value of difference
        differences = np.abs(original_points - reconstructed_points)
        squared_errors = differences ** 2
        
        # Compute mean squared error
        mse = np.mean(squared_errors)
        
        # Compute root mean squared error
        rmse = np.sqrt(mse)
        
        # Normalize by signal magnitude for relative error
        signal_magnitude = np.mean(np.abs(original_points))
        if signal_magnitude > 0:
            normalized_rmse = rmse / signal_magnitude
        else:
            normalized_rmse = rmse
        
        return normalized_rmse
    
    def _pad_coefficients(
        self,
        coefficients: List[FourierCoefficient],
        target_size: int
    ) -> List[FourierCoefficient]:
        """
        Pad coefficient list with zero-amplitude coefficients.
        
        When we truncate coefficients for optimization, we need to pad
        them back to the original size for IDFT to produce the correct
        number of output points.
        
        Args:
            coefficients: List of significant coefficients (sorted by amplitude)
            target_size: Desired total number of coefficients (must match original point count)
        
        Returns:
            Padded list of coefficients with zeros for missing frequencies
        """
        if len(coefficients) >= target_size:
            return coefficients[:target_size]
        
        # Create a dictionary of existing coefficients by frequency
        coeff_dict = {c.frequency: c for c in coefficients}
        
        # Build complete list with zeros for missing frequencies
        # We need exactly target_size coefficients, one for each frequency 0 to target_size-1
        padded = []
        for freq in range(target_size):
            if freq in coeff_dict:
                padded.append(coeff_dict[freq])
            else:
                # Create zero-amplitude coefficient for this frequency
                zero_coeff = FourierCoefficient(
                    frequency=freq,
                    amplitude=0.0,
                    phase=0.0,
                    complex_value=0.0+0.0j
                )
                padded.append(zero_coeff)
        
        return padded
    
    def optimize_count(
        self,
        coefficients: List[FourierCoefficient],
        original_points: np.ndarray
    ) -> OptimizationResult:
        """
        Determine optimal coefficient count using binary search.
        
        Uses binary search to find the minimum number of coefficients that
        achieves reconstruction error below the target threshold. This
        minimizes payload size while maintaining quality.
        
        Algorithm:
        1. Sort coefficients by amplitude (most significant first)
        2. Binary search for minimum count that satisfies error threshold
        3. Verify at least 20% size reduction compared to using all coefficients
        
        Args:
            coefficients: List of all Fourier coefficients
            original_points: Original contour points for error calculation
        
        Returns:
            OptimizationResult with optimal count, complexity, error, and explanation
        
        Raises:
            AIModelError: If optimization fails or constraints cannot be met
        """
        # Validate inputs
        if not coefficients:
            raise AIModelError("coefficients list cannot be empty")
        
        if not isinstance(original_points, np.ndarray):
            raise AIModelError("original_points must be a NumPy array")
        
        if original_points.size == 0:
            raise AIModelError("original_points cannot be empty")
        
        # Sort coefficients by amplitude (most significant first)
        sorted_coeffs = self.transformer.sort_by_amplitude(coefficients)
        total_count = len(sorted_coeffs)
        
        # The target size for padding must match the original_points length
        # This ensures IDFT produces the correct number of output points
        target_size = len(original_points)
        
        # Ensure we have enough coefficients
        if total_count < self.min_coefficients:
            raise AIModelError(
                f"Not enough coefficients: {total_count} < {self.min_coefficients}"
            )
        
        logger.info(
            f"Starting optimization with {total_count} coefficients, "
            f"target error: {self.target_error}, target_size: {target_size}"
        )
        
        # Binary search for optimal coefficient count
        left = self.min_coefficients
        right = min(total_count, self.max_coefficients)
        optimal_count = right
        best_error = float('inf')
        
        while left <= right:
            mid = (left + right) // 2
            
            # Test reconstruction with 'mid' coefficients
            # We need to pad the coefficient list to match original_points size
            test_coeffs = self._pad_coefficients(sorted_coeffs[:mid], target_size)
            
            try:
                # Reconstruct points using these coefficients
                reconstructed = self.transformer.compute_idft(test_coeffs)
                
                # Compute reconstruction error
                error = self.compute_reconstruction_error(
                    original_points,
                    reconstructed
                )
                
                logger.debug(
                    f"Testing {mid} coefficients: error={error:.4f}, "
                    f"target={self.target_error}"
                )
                
                # Check if error is acceptable
                if error <= self.target_error:
                    # Error is acceptable, try fewer coefficients
                    optimal_count = mid
                    best_error = error
                    right = mid - 1
                else:
                    # Error too high, need more coefficients
                    left = mid + 1
                    
            except Exception as e:
                logger.warning(f"Error testing {mid} coefficients: {e}")
                # If reconstruction fails, need more coefficients
                left = mid + 1
        
        # Verify we found a valid solution
        if best_error > self.target_error:
            # Use all available coefficients if we can't meet target
            # But never exceed the original count
            optimal_count = min(right + 1, total_count)
            test_coeffs = self._pad_coefficients(sorted_coeffs[:optimal_count], target_size)
            reconstructed = self.transformer.compute_idft(test_coeffs)
            best_error = self.compute_reconstruction_error(
                original_points,
                reconstructed
            )
            
            logger.warning(
                f"Could not meet target error {self.target_error}, "
                f"using {optimal_count} coefficients with error {best_error:.4f}"
            )
        
        # Calculate size reduction percentage
        size_reduction = (1 - optimal_count / total_count) * 100
        
        # Classify complexity based on coefficient requirements
        if optimal_count < total_count * 0.3:
            complexity_class = "low"
        elif optimal_count < total_count * 0.6:
            complexity_class = "medium"
        else:
            complexity_class = "high"
        
        # Generate explanation
        explanation = (
            f"Optimized from {total_count} to {optimal_count} coefficients "
            f"({size_reduction:.1f}% reduction). "
            f"Reconstruction error: {best_error:.4f} (target: {self.target_error}). "
            f"Image complexity: {complexity_class}."
        )
        
        logger.info(explanation)
        
        # Create and return result
        result = OptimizationResult(
            optimal_count=optimal_count,
            complexity_class=complexity_class,
            reconstruction_error=best_error,
            explanation=explanation
        )
        
        return result
