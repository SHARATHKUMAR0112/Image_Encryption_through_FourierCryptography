"""
Core data models for Fourier-Based Image Encryption System.

This module defines immutable dataclasses for representing Fourier coefficients,
contours, epicycle states, encrypted payloads, and AI-related results.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class FourierCoefficient:
    """
    Immutable Fourier coefficient representation.
    
    Each coefficient represents a rotating vector (epicycle) in the complex plane
    with a specific frequency, amplitude (radius), and phase (initial angle).
    
    Attributes:
        frequency: Integer frequency index k in DFT formula
        amplitude: Magnitude |F(k)| - radius of the epicycle (must be non-negative)
        phase: Argument arg(F(k)) - initial angle in radians (must be in [-π, π])
        complex_value: Complex number F(k) = amplitude * e^(i*phase)
    """
    frequency: int
    amplitude: float
    phase: float
    complex_value: complex
    
    def __post_init__(self):
        """Validate coefficient fields after initialization."""
        # Validate amplitude is non-negative
        if self.amplitude < 0:
            raise ValueError(
                f"Amplitude must be non-negative, got {self.amplitude}"
            )
        
        # Validate phase is in [-π, π]
        if not (-math.pi <= self.phase <= math.pi):
            raise ValueError(
                f"Phase must be in [-π, π], got {self.phase}"
            )
        
        # Validate complex_value is consistent with amplitude and phase
        expected_complex = self.amplitude * complex(
            math.cos(self.phase), math.sin(self.phase)
        )
        if not math.isclose(abs(self.complex_value - expected_complex), 0, abs_tol=1e-9):
            raise ValueError(
                f"Complex value {self.complex_value} is inconsistent with "
                f"amplitude {self.amplitude} and phase {self.phase}"
            )


@dataclass
class Contour:
    """
    Contour representation extracted from image edges.
    
    Attributes:
        points: Nx2 NumPy array of (x, y) coordinates
        is_closed: Whether the contour forms a closed loop
        length: Number of points in the contour
    """
    points: np.ndarray
    is_closed: bool
    length: int
    
    def __post_init__(self):
        """Validate contour data after initialization."""
        if not isinstance(self.points, np.ndarray):
            raise TypeError("points must be a NumPy array")
        
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(
                f"points must be Nx2 array, got shape {self.points.shape}"
            )
        
        if self.length != len(self.points):
            raise ValueError(
                f"length {self.length} does not match points array length {len(self.points)}"
            )


@dataclass
class EpicycleState:
    """
    State of epicycle animation at a specific time t.
    
    Represents the positions of all epicycles and the trace point being drawn
    at a particular moment in the animation.
    
    Attributes:
        time: Current time parameter t in [0, 2π]
        positions: List of complex numbers representing each epicycle center position
        trace_point: Final point being drawn (tip of the last epicycle)
    """
    time: float
    positions: List[complex]
    trace_point: complex
    
    def __post_init__(self):
        """Validate epicycle state after initialization."""
        if not (0 <= self.time <= 2 * math.pi):
            raise ValueError(
                f"time must be in [0, 2π], got {self.time}"
            )
        
        if not self.positions:
            raise ValueError("positions list cannot be empty")


@dataclass
class EncryptedPayload:
    """
    Complete encrypted package containing ciphertext and authentication data.
    
    Attributes:
        ciphertext: Encrypted coefficient data
        iv: Initialization vector (16 bytes for AES)
        hmac: HMAC-SHA256 authentication tag (32 bytes)
        metadata: Additional information (version, coefficient count, dimensions)
    """
    ciphertext: bytes
    iv: bytes
    hmac: bytes
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate encrypted payload after initialization."""
        if not isinstance(self.ciphertext, bytes):
            raise TypeError("ciphertext must be bytes")
        
        if not isinstance(self.iv, bytes):
            raise TypeError("iv must be bytes")
        
        if len(self.iv) != 16:
            raise ValueError(
                f"iv must be 16 bytes for AES, got {len(self.iv)} bytes"
            )
        
        if not isinstance(self.hmac, bytes):
            raise TypeError("hmac must be bytes")
        
        if len(self.hmac) != 32:
            raise ValueError(
                f"hmac must be 32 bytes for SHA256, got {len(self.hmac)} bytes"
            )
        
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")


@dataclass
class OptimizationResult:
    """
    Result from AI coefficient optimizer.
    
    Contains the optimal number of coefficients determined by the AI optimizer,
    along with complexity classification and reconstruction error metrics.
    
    Attributes:
        optimal_count: Optimal number of Fourier coefficients
        complexity_class: Image complexity classification ("low", "medium", "high")
        reconstruction_error: RMSE between original and reconstructed contour
        explanation: Human-readable explanation of why this count was chosen
    """
    optimal_count: int
    complexity_class: str
    reconstruction_error: float
    explanation: str
    
    def __post_init__(self):
        """Validate optimization result after initialization."""
        if not (10 <= self.optimal_count <= 1000):
            raise ValueError(
                f"optimal_count must be in [10, 1000], got {self.optimal_count}"
            )
        
        valid_classes = {"low", "medium", "high"}
        if self.complexity_class not in valid_classes:
            raise ValueError(
                f"complexity_class must be one of {valid_classes}, "
                f"got '{self.complexity_class}'"
            )
        
        if self.reconstruction_error < 0:
            raise ValueError(
                f"reconstruction_error must be non-negative, "
                f"got {self.reconstruction_error}"
            )
        
        if not self.explanation or not self.explanation.strip():
            raise ValueError("explanation cannot be empty")


@dataclass
class AnomalyReport:
    """
    Report from AI anomaly detector.
    
    Contains information about detected anomalies in encrypted coefficients,
    including confidence, type, and severity.
    
    Attributes:
        is_anomalous: Whether an anomaly was detected
        confidence: Detection confidence score in [0, 1]
        anomaly_type: Type of anomaly ("tampered", "corrupted", "none")
        severity: Severity level ("low", "medium", "high", "critical")
        details: Human-readable details about the anomaly
    """
    is_anomalous: bool
    confidence: float
    anomaly_type: str
    severity: str
    details: str
    
    def __post_init__(self):
        """Validate anomaly report after initialization."""
        if not (0 <= self.confidence <= 1):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        
        valid_types = {"tampered", "corrupted", "none"}
        if self.anomaly_type not in valid_types:
            raise ValueError(
                f"anomaly_type must be one of {valid_types}, "
                f"got '{self.anomaly_type}'"
            )
        
        valid_severities = {"low", "medium", "high", "critical"}
        if self.severity not in valid_severities:
            raise ValueError(
                f"severity must be one of {valid_severities}, "
                f"got '{self.severity}'"
            )


@dataclass
class Metrics:
    """
    Real-time metrics for monitoring dashboard.
    
    Tracks performance and progress metrics during encryption/decryption operations.
    
    Attributes:
        current_coefficient_index: Index of coefficient currently being processed
        active_radius: Radius magnitude of the active epicycle
        progress_percentage: Overall progress percentage in [0, 100]
        fps: Current frames per second for animation
        processing_time: Total processing time in seconds
        memory_usage_mb: Current memory usage in megabytes
        encryption_status: Current status ("idle", "encrypting", "decrypting", "complete", "error")
    """
    current_coefficient_index: int
    active_radius: float
    progress_percentage: float
    fps: float
    processing_time: float
    memory_usage_mb: float
    encryption_status: str
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if self.current_coefficient_index < 0:
            raise ValueError(
                f"current_coefficient_index must be non-negative, "
                f"got {self.current_coefficient_index}"
            )
        
        if self.active_radius < 0:
            raise ValueError(
                f"active_radius must be non-negative, got {self.active_radius}"
            )
        
        if not (0 <= self.progress_percentage <= 100):
            raise ValueError(
                f"progress_percentage must be in [0, 100], "
                f"got {self.progress_percentage}"
            )
        
        if self.fps < 0:
            raise ValueError(f"fps must be non-negative, got {self.fps}")
        
        if self.processing_time < 0:
            raise ValueError(
                f"processing_time must be non-negative, got {self.processing_time}"
            )
        
        if self.memory_usage_mb < 0:
            raise ValueError(
                f"memory_usage_mb must be non-negative, got {self.memory_usage_mb}"
            )
        
        valid_statuses = {"idle", "encrypting", "decrypting", "complete", "error"}
        if self.encryption_status not in valid_statuses:
            raise ValueError(
                f"encryption_status must be one of {valid_statuses}, "
                f"got '{self.encryption_status}'"
            )


@dataclass
class EdgeDetectionConfig:
    """
    Configuration for industrial edge detection pipeline.
    
    Provides tunable parameters for the IndustrialEdgeDetector, including
    GrabCut foreground extraction, Gaussian preprocessing, Canny edge detection,
    and morphological refinement settings.
    
    Attributes:
        grabcut_iterations: Number of GrabCut iterations for foreground extraction (default: 5)
        canny_threshold1: Lower threshold for Canny edge detection (default: 50)
        canny_threshold2: Upper threshold for Canny edge detection (default: 150)
        gaussian_kernel: Kernel size for Gaussian blur preprocessing (default: 5, must be odd)
        morph_kernel_size: Kernel size for morphological operations (default: 3)
        enable_morphology: Whether to apply morphological refinement (default: True)
        enable_foreground_extraction: Whether to apply GrabCut foreground extraction (default: True)
    """
    grabcut_iterations: int = 5
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    gaussian_kernel: int = 5
    morph_kernel_size: int = 3
    enable_morphology: bool = True
    enable_foreground_extraction: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.grabcut_iterations < 1:
            raise ValueError(
                f"grabcut_iterations must be at least 1, got {self.grabcut_iterations}"
            )
        
        if not (0 <= self.canny_threshold1 <= 255):
            raise ValueError(
                f"canny_threshold1 must be in [0, 255], got {self.canny_threshold1}"
            )
        
        if not (0 <= self.canny_threshold2 <= 255):
            raise ValueError(
                f"canny_threshold2 must be in [0, 255], got {self.canny_threshold2}"
            )
        
        if self.canny_threshold1 >= self.canny_threshold2:
            raise ValueError(
                f"canny_threshold1 ({self.canny_threshold1}) must be less than "
                f"canny_threshold2 ({self.canny_threshold2})"
            )
        
        if self.gaussian_kernel < 1 or self.gaussian_kernel % 2 == 0:
            raise ValueError(
                f"gaussian_kernel must be a positive odd number, got {self.gaussian_kernel}"
            )
        
        if self.morph_kernel_size < 1:
            raise ValueError(
                f"morph_kernel_size must be at least 1, got {self.morph_kernel_size}"
            )


@dataclass
class ReconstructionConfig:
    """
    Configuration for image reconstruction from Fourier coefficients.
    
    Controls reconstruction mode (static vs animated), animation speed,
    quality settings, and output format options.
    
    Attributes:
        mode: Reconstruction mode - "static" (final image only) or "animated" (epicycle drawing process)
        speed: Animation speed multiplier in [0.1, 10.0] (default: 1.0)
        quality: Quality mode - "fast" (30 frames), "balanced" (100 frames), or "quality" (300 frames)
        save_frames: Whether to save individual frames as PNG sequence (default: False)
        save_animation: Whether to save animation as video/GIF file (default: False)
        output_format: Output format - "mp4", "gif", or "png_sequence" (default: "mp4")
        output_path: Path for saving output files (optional)
        backend: Visualization backend - "pyqtgraph" or "matplotlib" (default: "pyqtgraph")
    """
    mode: str = "animated"
    speed: float = 1.0
    quality: str = "balanced"
    save_frames: bool = False
    save_animation: bool = False
    output_format: str = "mp4"
    output_path: Optional[str] = None
    backend: str = "pyqtgraph"
    
    def __post_init__(self):
        """Validate reconstruction configuration parameters."""
        valid_modes = {"static", "animated"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}, got '{self.mode}'"
            )
        
        if not (0.1 <= self.speed <= 10.0):
            raise ValueError(
                f"speed must be in [0.1, 10.0], got {self.speed}"
            )
        
        valid_qualities = {"fast", "balanced", "quality"}
        if self.quality not in valid_qualities:
            raise ValueError(
                f"quality must be one of {valid_qualities}, got '{self.quality}'"
            )
        
        valid_formats = {"mp4", "gif", "png_sequence"}
        if self.output_format not in valid_formats:
            raise ValueError(
                f"output_format must be one of {valid_formats}, got '{self.output_format}'"
            )
        
        valid_backends = {"pyqtgraph", "matplotlib"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got '{self.backend}'"
            )
    
    def get_frame_count(self) -> int:
        """
        Return frame count based on quality mode.
        
        Returns:
            30 for "fast", 100 for "balanced", 300 for "quality"
        """
        quality_to_frames = {
            "fast": 30,
            "balanced": 100,
            "quality": 300
        }
        return quality_to_frames[self.quality]


@dataclass
class ReconstructionResult:
    """
    Result of image reconstruction operation.
    
    Contains the final reconstructed image, optional animation frames,
    and metadata about the reconstruction process.
    
    Attributes:
        final_image: Final reconstructed image as NumPy array
        frames: List of animation frames (if saved), None otherwise
        animation_path: Path to saved animation file (if saved), None otherwise
        reconstruction_time: Time taken for reconstruction in seconds
        frame_count: Number of frames generated during reconstruction
    """
    final_image: np.ndarray
    frames: Optional[List[np.ndarray]] = None
    animation_path: Optional[str] = None
    reconstruction_time: float = 0.0
    frame_count: int = 0
    
    def __post_init__(self):
        """Validate reconstruction result after initialization."""
        if not isinstance(self.final_image, np.ndarray):
            raise TypeError("final_image must be a NumPy array")
        
        if self.final_image.size == 0:
            raise ValueError("final_image cannot be empty")
        
        if self.frames is not None:
            if not isinstance(self.frames, list):
                raise TypeError("frames must be a list or None")
            
            if not all(isinstance(frame, np.ndarray) for frame in self.frames):
                raise TypeError("all frames must be NumPy arrays")
        
        if self.reconstruction_time < 0:
            raise ValueError(
                f"reconstruction_time must be non-negative, got {self.reconstruction_time}"
            )
        
        if self.frame_count < 0:
            raise ValueError(
                f"frame_count must be non-negative, got {self.frame_count}"
            )
