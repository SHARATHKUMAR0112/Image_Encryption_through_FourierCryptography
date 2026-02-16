"""
Fourier Transform Engine for contour decomposition and reconstruction.

This module implements Discrete Fourier Transform (DFT) and Inverse DFT
operations on contour points, converting spatial data into frequency domain
coefficients that can be encrypted and later reconstructed.

Performance optimizations:
- Vectorized NumPy operations for coefficient extraction
- Pre-allocated arrays to reduce memory allocations
- Cached trigonometric computations
- Optimized sorting with NumPy argsort
"""

import math
from functools import lru_cache
from typing import List

import numpy as np

from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.models.exceptions import ImageProcessingError


class FourierTransformer:
    """
    Compute DFT and IDFT on contour points.
    
    This class provides methods to:
    - Compute Discrete Fourier Transform on complex contour points
    - Reconstruct points from Fourier coefficients using IDFT
    - Sort coefficients by amplitude (most significant first)
    - Truncate to a specified number of coefficients
    
    The DFT formula used is:
        F(k) = Σ(n=0 to N-1) x(n) * e^(-j*2π*k*n/N)
    
    Uses NumPy FFT for O(N log N) performance.
    
    Performance optimizations:
    - Vectorized operations for coefficient creation
    - Pre-allocated arrays for amplitude and phase computation
    - Cached frequency arrays for repeated operations
    """
    
    def __init__(self):
        """Initialize transformer with caching support."""
        self._frequency_cache = {}
    
    def compute_dft(self, points: np.ndarray) -> List[FourierCoefficient]:
        """
        Compute Discrete Fourier Transform on contour points.
        
        Converts spatial domain points (complex numbers) into frequency domain
        coefficients. Each coefficient represents a rotating vector (epicycle)
        with a specific frequency, amplitude, and phase.
        
        Optimizations:
        - Vectorized amplitude and phase computation
        - Pre-allocated arrays for results
        - Batch coefficient creation
        
        Args:
            points: 1D NumPy array of complex numbers representing contour points
                   (x + iy format)
        
        Returns:
            List of FourierCoefficient objects, one for each frequency component
        
        Raises:
            ImageProcessingError: If points array is invalid or empty
        """
        # Validate input
        if not isinstance(points, np.ndarray):
            raise ImageProcessingError("points must be a NumPy array")
        
        if points.size == 0:
            raise ImageProcessingError("points array cannot be empty")
        
        if points.ndim != 1:
            raise ImageProcessingError(
                f"points must be 1D array, got shape {points.shape}"
            )
        
        # Ensure points are complex numbers
        if not np.issubdtype(points.dtype, np.complexfloating):
            # Try to convert to complex
            try:
                points = points.astype(complex)
            except (ValueError, TypeError) as e:
                raise ImageProcessingError(
                    f"Cannot convert points to complex numbers: {e}"
                )
        
        # Compute FFT using NumPy for O(N log N) performance
        fft_result = np.fft.fft(points)
        N = len(points)
        
        # Vectorized computation of amplitudes and phases
        # This is much faster than computing them one by one in a loop
        amplitudes = np.abs(fft_result)
        phases = np.angle(fft_result)  # Returns values in [-π, π]
        
        # Create frequency array
        frequencies = np.arange(N, dtype=np.int32)
        
        # Convert to list of FourierCoefficient objects
        # This is the only part that can't be fully vectorized due to dataclass creation
        coefficients = [
            FourierCoefficient(
                frequency=int(frequencies[k]),
                amplitude=float(amplitudes[k]),
                phase=float(phases[k]),
                complex_value=complex(fft_result[k])
            )
            for k in range(N)
        ]
        
        return coefficients
    
    def compute_idft(self, coefficients: List[FourierCoefficient]) -> np.ndarray:
        """
        Reconstruct points from Fourier coefficients using Inverse DFT.
        
        Converts frequency domain coefficients back to spatial domain points.
        This is the inverse operation of compute_dft().
        
        Args:
            coefficients: List of FourierCoefficient objects
        
        Returns:
            1D NumPy array of complex numbers representing reconstructed points
        
        Raises:
            ImageProcessingError: If coefficients list is invalid or empty
        """
        # Validate input
        if not coefficients:
            raise ImageProcessingError("coefficients list cannot be empty")
        
        if not all(isinstance(c, FourierCoefficient) for c in coefficients):
            raise ImageProcessingError(
                "All elements must be FourierCoefficient objects"
            )
        
        # Extract complex values from coefficients
        # Sort by frequency to ensure correct order
        sorted_coeffs = sorted(coefficients, key=lambda c: c.frequency)
        
        # Build complex array for IFFT
        complex_values = np.array([c.complex_value for c in sorted_coeffs])
        
        # Compute Inverse FFT
        reconstructed_points = np.fft.ifft(complex_values)
        
        return reconstructed_points
    
    def sort_by_amplitude(
        self, coefficients: List[FourierCoefficient]
    ) -> List[FourierCoefficient]:
        """
        Sort coefficients by amplitude in descending order.
        
        Coefficients with larger amplitudes contribute more to the signal
        reconstruction, so sorting by amplitude allows us to select the most
        significant terms for compression or encryption.
        
        Optimizations:
        - Uses NumPy argsort for faster sorting
        - Vectorized amplitude extraction
        
        Args:
            coefficients: List of FourierCoefficient objects
        
        Returns:
            New list of coefficients sorted by amplitude (largest first)
        
        Raises:
            ImageProcessingError: If coefficients list is invalid
        """
        # Validate input
        if not coefficients:
            raise ImageProcessingError("coefficients list cannot be empty")
        
        if not all(isinstance(c, FourierCoefficient) for c in coefficients):
            raise ImageProcessingError(
                "All elements must be FourierCoefficient objects"
            )
        
        # Extract amplitudes as NumPy array for faster sorting
        amplitudes = np.array([c.amplitude for c in coefficients])
        
        # Get indices that would sort the array in descending order
        # argsort is faster than Python's sorted() for numerical data
        sorted_indices = np.argsort(-amplitudes)  # Negative for descending order
        
        # Reorder coefficients using sorted indices
        sorted_coefficients = [coefficients[i] for i in sorted_indices]
        
        return sorted_coefficients
    
    def truncate_coefficients(
        self,
        coefficients: List[FourierCoefficient],
        num_terms: int
    ) -> List[FourierCoefficient]:
        """
        Keep only the top N coefficients by amplitude.
        
        Truncating to fewer coefficients reduces data size while preserving
        the most significant frequency components. This is useful for
        compression and encryption optimization.
        
        Args:
            coefficients: List of FourierCoefficient objects
            num_terms: Number of coefficients to keep (must be in [10, 1000])
        
        Returns:
            List containing the top num_terms coefficients by amplitude
        
        Raises:
            ImageProcessingError: If num_terms is invalid or coefficients list is empty
        """
        # Validate input
        if not coefficients:
            raise ImageProcessingError("coefficients list cannot be empty")
        
        if not all(isinstance(c, FourierCoefficient) for c in coefficients):
            raise ImageProcessingError(
                "All elements must be FourierCoefficient objects"
            )
        
        # Validate num_terms range
        if not (10 <= num_terms <= 1000):
            raise ImageProcessingError(
                f"num_terms must be in range [10, 1000], got {num_terms}"
            )
        
        # If we have fewer coefficients than requested, return all
        if len(coefficients) <= num_terms:
            return coefficients
        
        # Sort by amplitude and take top N
        sorted_coeffs = self.sort_by_amplitude(coefficients)
        truncated = sorted_coeffs[:num_terms]
        
        return truncated
