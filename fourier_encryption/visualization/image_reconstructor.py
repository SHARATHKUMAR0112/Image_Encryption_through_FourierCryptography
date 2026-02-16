"""
Image reconstruction module for visualizing Fourier-based decryption.

This module provides the ImageReconstructor class that reconstructs images
from decrypted Fourier coefficients using epicycle visualization. Supports
both static reconstruction (final image only) and animated reconstruction
(epicycle drawing process).
"""

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.models.data_models import (
    FourierCoefficient,
    ReconstructionConfig,
    ReconstructionResult,
)
from fourier_encryption.models.exceptions import ConfigurationError
from fourier_encryption.visualization.live_renderer import LiveRenderer


class ImageReconstructor:
    """
    Dedicated module for reconstructing images from Fourier coefficients.
    
    Supports both static reconstruction (final image only) and animated
    reconstruction (epicycle drawing process). Integrates with existing
    epicycle_engine.py and visualization backends.
    
    Attributes:
        config: ReconstructionConfig specifying reconstruction parameters
        renderer: LiveRenderer for visualization (if animated mode)
        transformer: FourierTransformer for IDFT computation
    """
    
    def __init__(self, config: ReconstructionConfig):
        """
        Initialize the image reconstructor with configuration.
        
        Args:
            config: ReconstructionConfig specifying reconstruction parameters
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config
        self.renderer = None
        self.transformer = FourierTransformer()
        
        # Initialize renderer only for animated mode
        if config.mode == "animated":
            self.renderer = LiveRenderer(backend=config.backend)
    
    def reconstruct_static(self, coefficients: List[FourierCoefficient]) -> np.ndarray:
        """
        Perform static reconstruction - return final image only.
        
        Uses IDFT to directly compute final contour points without animation.
        Fast mode for when visualization is not needed.
        
        Args:
            coefficients: List of Fourier coefficients to reconstruct from
            
        Returns:
            Final reconstructed image as NumPy array
            
        Raises:
            ValueError: If coefficients list is empty
        """
        if not coefficients:
            raise ValueError("coefficients list cannot be empty")
        
        # Use IDFT to directly compute final contour points
        reconstructed_points = self.transformer.compute_idft(coefficients)
        
        # Convert complex points to 2D coordinates
        points_2d = np.column_stack([
            reconstructed_points.real,
            reconstructed_points.imag
        ])
        
        # Create image from points
        image = self._points_to_image(points_2d)
        
        return image
    
    def reconstruct_animated(
        self,
        coefficients: List[FourierCoefficient],
        engine: EpicycleEngine
    ) -> ReconstructionResult:
        """
        Perform animated reconstruction - generate epicycle drawing animation.
        
        Uses EpicycleEngine to compute epicycle positions at each frame.
        Renders animation using configured backend (PyQtGraph or Matplotlib).
        Optionally saves frames or complete animation.
        
        Args:
            coefficients: List of Fourier coefficients to reconstruct from
            engine: EpicycleEngine for computing epicycle positions
            
        Returns:
            ReconstructionResult with final image, frames, and metadata
            
        Raises:
            ValueError: If coefficients list is empty
            ConfigurationError: If renderer is not initialized
        """
        if not coefficients:
            raise ValueError("coefficients list cannot be empty")
        
        if self.renderer is None:
            raise ConfigurationError(
                "LiveRenderer not initialized for animated reconstruction"
            )
        
        start_time = time.time()
        
        # Get frame count based on quality mode
        num_frames = self.config.get_frame_count()
        
        # Generate animation frames
        frames = []
        trace_points = []
        
        for frame_num, state in enumerate(
            engine.generate_animation_frames(num_frames, self.config.speed)
        ):
            # Render the frame
            self.renderer.render_frame(state, frame_num)
            
            # Collect trace point
            trace_points.append(state.trace_point)
            
            # Capture frame if saving
            if self.config.save_frames or self.config.save_animation:
                frame_image = self.render_frame_to_image(state)
                frames.append(frame_image)
        
        # Create final image from trace points
        trace_points_2d = np.column_stack([
            np.array([p.real for p in trace_points]),
            np.array([p.imag for p in trace_points])
        ])
        final_image = self._points_to_image(trace_points_2d)
        
        reconstruction_time = time.time() - start_time
        
        # Save frames if requested
        animation_path = None
        if self.config.save_frames and self.config.output_path:
            output_dir = Path(self.config.output_path)
            self.save_frames(frames, output_dir)
        
        # Save animation if requested
        if self.config.save_animation and self.config.output_path:
            output_path = Path(self.config.output_path)
            if self.config.output_format == "mp4":
                animation_path = str(output_path.with_suffix(".mp4"))
                self.save_animation_video(frames, Path(animation_path))
            elif self.config.output_format == "gif":
                animation_path = str(output_path.with_suffix(".gif"))
                self.save_animation_gif(frames, Path(animation_path))
            elif self.config.output_format == "png_sequence":
                animation_path = str(output_path)
                self.save_frames(frames, Path(animation_path))
        
        return ReconstructionResult(
            final_image=final_image,
            frames=frames if (self.config.save_frames or self.config.save_animation) else None,
            animation_path=animation_path,
            reconstruction_time=reconstruction_time,
            frame_count=num_frames
        )
    
    def save_frames(self, frames: List[np.ndarray], output_dir: Path) -> None:
        """
        Save individual frames as PNG sequence.
        
        Args:
            frames: List of frame images as NumPy arrays
            output_dir: Directory to save frames in
            
        Raises:
            IOError: If unable to create directory or save frames
        """
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each frame
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
    
    def save_animation_video(self, frames: List[np.ndarray], output_path: Path) -> None:
        """
        Save frames as MP4 video using OpenCV VideoWriter.
        
        Args:
            frames: List of frame images as NumPy arrays
            output_path: Path to save MP4 video
            
        Raises:
            IOError: If unable to create video file
        """
        if not frames:
            raise ValueError("frames list cannot be empty")
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Standard 30 FPS
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        # Write frames
        for frame in frames:
            # Ensure frame is in BGR format for OpenCV
            if len(frame.shape) == 2:  # Grayscale
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame
            
            video_writer.write(frame_bgr)
        
        video_writer.release()
    
    def save_animation_gif(self, frames: List[np.ndarray], output_path: Path) -> None:
        """
        Save frames as animated GIF using PIL.
        
        Args:
            frames: List of frame images as NumPy arrays
            output_path: Path to save animated GIF
            
        Raises:
            IOError: If unable to create GIF file
        """
        if not frames:
            raise ValueError("frames list cannot be empty")
        
        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            pil_frame = Image.fromarray(frame_rgb)
            pil_frames.append(pil_frame)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as animated GIF
        pil_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=33,  # ~30 FPS (1000ms / 30 = 33ms per frame)
            loop=0  # Infinite loop
        )
    
    def render_frame_to_image(self, state) -> np.ndarray:
        """
        Render a single epicycle state to an image array.
        
        Converts visualization backend output to numpy array for saving.
        This is a simplified implementation that creates a basic visualization.
        
        Args:
            state: EpicycleState to render
            
        Returns:
            Rendered frame as NumPy array
        """
        # Create a blank image
        img_size = 800
        image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Find bounds for scaling
        all_points = [state.trace_point] + state.positions
        real_parts = [p.real for p in all_points]
        imag_parts = [p.imag for p in all_points]
        
        if real_parts and imag_parts:
            min_x, max_x = min(real_parts), max(real_parts)
            min_y, max_y = min(imag_parts), max(imag_parts)
            
            # Add padding
            padding = 50
            range_x = max_x - min_x if max_x != min_x else 1
            range_y = max_y - min_y if max_y != min_y else 1
            scale = min((img_size - 2 * padding) / range_x, (img_size - 2 * padding) / range_y)
            
            # Transform function
            def transform(p: complex) -> tuple:
                x = int((p.real - min_x) * scale + padding)
                y = int((p.imag - min_y) * scale + padding)
                return (x, y)
            
            # Draw epicycle circles and connections
            prev_pos = complex(0, 0)
            for i, pos in enumerate(state.positions):
                # Draw line from previous position to current
                pt1 = transform(prev_pos)
                pt2 = transform(pos)
                cv2.line(image, pt1, pt2, (100, 100, 100), 1)
                prev_pos = pos
            
            # Draw trace point
            trace_pt = transform(state.trace_point)
            cv2.circle(image, trace_pt, 3, (0, 255, 0), -1)
        
        return image
    
    def _points_to_image(self, points: np.ndarray) -> np.ndarray:
        """
        Convert 2D points to an image.
        
        Args:
            points: Nx2 array of (x, y) coordinates
            
        Returns:
            Image as NumPy array with points drawn
        """
        if len(points) == 0:
            # Return blank image if no points
            return np.zeros((800, 800), dtype=np.uint8)
        
        # Find bounds
        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        min_y, max_y = points[:, 1].min(), points[:, 1].max()
        
        # Add padding
        padding = 50
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1
        
        # Determine image size
        img_size = 800
        scale = min((img_size - 2 * padding) / range_x, (img_size - 2 * padding) / range_y)
        
        # Create blank image
        image = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # Transform and draw points
        for i in range(len(points) - 1):
            pt1 = (
                int((points[i, 0] - min_x) * scale + padding),
                int((points[i, 1] - min_y) * scale + padding)
            )
            pt2 = (
                int((points[i + 1, 0] - min_x) * scale + padding),
                int((points[i + 1, 1] - min_y) * scale + padding)
            )
            cv2.line(image, pt1, pt2, 255, 2)
        
        return image
