"""
Contour extraction and processing for Fourier-Based Image Encryption System.

This module provides functionality for extracting contours from edge maps,
converting them to complex plane representation, and resampling for uniform
point distribution.
"""

from typing import List

import cv2
import numpy as np
from scipy import interpolate

from fourier_encryption.models.data_models import Contour
from fourier_encryption.models.exceptions import ImageProcessingError


class ContourExtractor:
    """
    Extract and process contours from edge maps.
    
    Provides methods for:
    - Extracting contours from binary edge maps using OpenCV
    - Converting (x, y) coordinates to complex plane representation
    - Resampling contours for uniform point distribution
    """
    
    def extract_contours(self, edge_map: np.ndarray, min_contour_length: int = 10) -> List[Contour]:
        """
        Extract contours from binary edge map using OpenCV findContours.
        
        Args:
            edge_map: Binary edge map (0 or 255 values)
            min_contour_length: Minimum number of points for a valid contour
            
        Returns:
            List of Contour objects sorted by length (descending)
            
        Raises:
            ImageProcessingError: If edge map is invalid or contour extraction fails
        """
        if not isinstance(edge_map, np.ndarray):
            raise ImageProcessingError(
                "edge_map must be a NumPy array",
                context={"type": type(edge_map).__name__}
            )
        
        if edge_map.ndim != 2:
            raise ImageProcessingError(
                "edge_map must be a 2D array",
                context={"shape": edge_map.shape}
            )
        
        # Ensure edge map is binary (0 or 255)
        if edge_map.dtype != np.uint8:
            edge_map = edge_map.astype(np.uint8)
        
        # Check if edge map is completely empty
        if np.all(edge_map == 0):
            raise ImageProcessingError(
                "Edge map is completely empty (no edges detected)",
                context={"shape": edge_map.shape}
            )
        
        try:
            # Find contours using OpenCV
            # RETR_LIST retrieves all contours without hierarchy
            # CHAIN_APPROX_NONE stores all contour points (no approximation)
            contours_raw, _ = cv2.findContours(
                edge_map,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_NONE
            )
            
            if not contours_raw:
                raise ImageProcessingError(
                    "No contours found in edge map",
                    context={"edge_pixels": np.count_nonzero(edge_map)}
                )
            
            # Convert OpenCV contours to our Contour dataclass
            contours = []
            for contour_raw in contours_raw:
                # OpenCV contours have shape (N, 1, 2), reshape to (N, 2)
                points = contour_raw.reshape(-1, 2).astype(np.float64)
                
                # Filter out contours that are too short
                if len(points) < min_contour_length:
                    continue
                
                # Check if contour is closed (first and last points are close)
                is_closed = np.linalg.norm(points[0] - points[-1]) < 2.0
                
                contours.append(Contour(
                    points=points,
                    is_closed=is_closed,
                    length=len(points)
                ))
            
            if not contours:
                raise ImageProcessingError(
                    f"No contours with at least {min_contour_length} points found",
                    context={
                        "total_contours": len(contours_raw),
                        "min_length": min_contour_length
                    }
                )
            
            # Sort contours by length (descending) - longest contours first
            contours.sort(key=lambda c: c.length, reverse=True)
            
            return contours
            
        except cv2.error as e:
            raise ImageProcessingError(
                f"OpenCV error during contour extraction: {e}",
                context={"error": str(e)}
            )
    
    def to_complex_plane(self, contour: Contour) -> np.ndarray:
        """
        Convert (x, y) contour points to complex plane representation.
        
        Each point (x, y) is converted to a complex number: x + iy
        
        Args:
            contour: Contour with (x, y) points
            
        Returns:
            NumPy array of complex numbers representing the contour
            
        Raises:
            ImageProcessingError: If conversion fails
        """
        if not isinstance(contour, Contour):
            raise ImageProcessingError(
                "Input must be a Contour object",
                context={"type": type(contour).__name__}
            )
        
        try:
            # Convert (x, y) points to complex numbers: x + iy
            complex_points = contour.points[:, 0] + 1j * contour.points[:, 1]
            return complex_points
            
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to convert contour to complex plane: {e}",
                context={
                    "error": str(e),
                    "contour_length": contour.length
                }
            )
    
    def resample_contour(self, contour: Contour, num_points: int) -> Contour:
        """
        Resample contour to have uniform point distribution.
        
        Uses spline interpolation to create a new contour with the specified
        number of evenly distributed points along the original contour path.
        
        Args:
            contour: Original contour
            num_points: Desired number of points in resampled contour
            
        Returns:
            New Contour with uniformly distributed points
            
        Raises:
            ImageProcessingError: If resampling fails or num_points is invalid
        """
        if not isinstance(contour, Contour):
            raise ImageProcessingError(
                "Input must be a Contour object",
                context={"type": type(contour).__name__}
            )
        
        if num_points < 10:
            raise ImageProcessingError(
                "num_points must be at least 10",
                context={"num_points": num_points, "minimum": 10}
            )
        
        if num_points > 10000:
            raise ImageProcessingError(
                "num_points exceeds maximum allowed (10000)",
                context={"num_points": num_points, "maximum": 10000}
            )
        
        try:
            points = contour.points
            
            # If contour already has the desired number of points, return copy
            if len(points) == num_points:
                return Contour(
                    points=points.copy(),
                    is_closed=contour.is_closed,
                    length=num_points
                )
            
            # Calculate cumulative distance along the contour
            distances = np.zeros(len(points))
            for i in range(1, len(points)):
                distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
            
            total_distance = distances[-1]
            
            # Handle degenerate case where all points are the same
            if total_distance < 1e-6:
                # Return contour with repeated points
                return Contour(
                    points=np.tile(points[0], (num_points, 1)),
                    is_closed=contour.is_closed,
                    length=num_points
                )
            
            # Normalize distances to [0, 1]
            normalized_distances = distances / total_distance
            
            # Create interpolation functions for x and y coordinates
            # Use cubic spline for smooth interpolation
            # k=min(3, len(points)-1) ensures we don't exceed available points
            k = min(3, len(points) - 1)
            
            interp_x = interpolate.UnivariateSpline(
                normalized_distances,
                points[:, 0],
                k=k,
                s=0  # No smoothing, interpolate exactly through points
            )
            
            interp_y = interpolate.UnivariateSpline(
                normalized_distances,
                points[:, 1],
                k=k,
                s=0
            )
            
            # Generate evenly spaced parameter values
            if contour.is_closed:
                # For closed contours, don't duplicate the last point
                t_new = np.linspace(0, 1, num_points, endpoint=False)
            else:
                # For open contours, include both endpoints
                t_new = np.linspace(0, 1, num_points)
            
            # Interpolate new points
            x_new = interp_x(t_new)
            y_new = interp_y(t_new)
            
            # Combine into (N, 2) array
            resampled_points = np.column_stack([x_new, y_new])
            
            return Contour(
                points=resampled_points,
                is_closed=contour.is_closed,
                length=num_points
            )
            
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to resample contour: {e}",
                context={
                    "error": str(e),
                    "original_length": contour.length,
                    "target_length": num_points
                }
            )
