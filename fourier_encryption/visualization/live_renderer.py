"""
Real-time epicycle animation renderer using Observer pattern.

This module implements the LiveRenderer class that visualizes epicycle-based
Fourier series reconstruction with support for multiple rendering backends
(PyQtGraph and Matplotlib).
"""

import time
from abc import ABC, abstractmethod
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.models.data_models import EpicycleState


class RenderObserver(ABC):
    """
    Abstract base class for render observers.
    
    Observers are notified when new frames are rendered, allowing them
    to react to animation updates (e.g., update metrics, save frames).
    """
    
    @abstractmethod
    def on_frame_rendered(self, state: EpicycleState, frame_number: int) -> None:
        """
        Called when a new frame is rendered.
        
        Args:
            state: Current epicycle state
            frame_number: Sequential frame number (0-indexed)
        """
        pass


class LiveRenderer:
    """
    Real-time epicycle animation using Observer pattern.
    
    The LiveRenderer visualizes the epicycle animation by drawing:
    1. All epicycle circles (with decreasing opacity for smaller ones)
    2. Connecting lines between epicycle centers
    3. Trace path (accumulated points forming the sketch)
    4. Current drawing point (highlighted)
    
    Supports multiple backends (matplotlib, pyqtgraph) and maintains
    target FPS for smooth animation.
    
    Attributes:
        backend: Rendering backend ("matplotlib" or "pyqtgraph")
        observers: List of registered observers
        _trace_points: Accumulated trace points for drawing the path
        _fig: Matplotlib figure (if using matplotlib backend)
        _ax: Matplotlib axes (if using matplotlib backend)
        _animation: Matplotlib animation object
    """
    
    def __init__(self, backend: str = "matplotlib"):
        """
        Initialize the live renderer.
        
        Args:
            backend: Rendering backend ("matplotlib" or "pyqtgraph")
        
        Raises:
            ValueError: If backend is not supported
        """
        valid_backends = {"matplotlib", "pyqtgraph"}
        if backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got '{backend}'"
            )
        
        self.backend = backend
        self.observers: List[RenderObserver] = []
        self._trace_points: List[complex] = []
        
        # Matplotlib-specific attributes
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._animation: Optional[FuncAnimation] = None
        
        # View transformation (for zoom/pan)
        self._view_center = complex(0, 0)
        self._view_scale = 1.0
        
        # Frame info tracking
        self._current_frame: int = 0
        self._total_frames: int = 0
        self._frame_info_text: Optional[plt.Text] = None
    
    def attach_observer(self, observer: RenderObserver) -> None:
        """
        Register an observer for frame updates.
        
        Args:
            observer: RenderObserver instance to register
        """
        if observer not in self.observers:
            self.observers.append(observer)
    
    def detach_observer(self, observer: RenderObserver) -> None:
        """
        Unregister an observer.
        
        Args:
            observer: RenderObserver instance to unregister
        """
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self, state: EpicycleState, frame_number: int) -> None:
        """
        Notify all observers of a new frame.
        
        Args:
            state: Current epicycle state
            frame_number: Sequential frame number
        """
        for observer in self.observers:
            observer.on_frame_rendered(state, frame_number)
    
    def render_frame(self, state: EpicycleState, frame_number: int = 0) -> None:
        """
        Draw epicycles, circles, and trace path for a single frame.
        
        This method renders:
        1. All epicycle circles (decreasing opacity for smaller ones)
        2. Connecting lines between epicycle centers
        3. Trace path (accumulated points)
        4. Current drawing point (highlighted)
        
        Args:
            state: Current epicycle state to render
            frame_number: Sequential frame number for observer notification
        """
        # Update current frame number
        self._current_frame = frame_number
        
        # Add current trace point to accumulated path
        self._trace_points.append(state.trace_point)
        
        # Notify observers
        self.notify_observers(state, frame_number)
        
        # Backend-specific rendering
        if self.backend == "matplotlib":
            self._render_matplotlib(state)
        elif self.backend == "pyqtgraph":
            self._render_pyqtgraph(state)
    
    def _render_matplotlib(self, state: EpicycleState) -> None:
        """
        Render frame using Matplotlib backend.
        
        Args:
            state: Current epicycle state to render
        """
        if self._ax is None:
            return
        
        # Clear previous frame
        self._ax.clear()
        
        # Extract real and imaginary parts for plotting
        positions_real = [pos.real for pos in state.positions]
        positions_imag = [pos.imag for pos in state.positions]
        
        # Draw connecting lines between epicycle centers
        if len(state.positions) > 1:
            self._ax.plot(
                positions_real,
                positions_imag,
                'b-',
                linewidth=1,
                alpha=0.5,
                label='Epicycle connections'
            )
        
        # Draw epicycle circles with decreasing opacity
        # Calculate amplitudes from consecutive positions
        for i in range(len(state.positions)):
            center = state.positions[i]
            
            # Calculate radius (distance to next position or trace point)
            if i < len(state.positions) - 1:
                next_pos = state.positions[i + 1]
                radius = abs(next_pos - center)
            else:
                radius = abs(state.trace_point - center)
            
            # Opacity decreases for smaller epicycles
            # Normalize by maximum radius for consistent opacity scaling
            max_radius = max(abs(state.positions[j+1] - state.positions[j]) 
                           for j in range(len(state.positions) - 1)) if len(state.positions) > 1 else 1.0
            if max_radius == 0:
                max_radius = 1.0
            
            opacity = 0.3 + 0.5 * (radius / max_radius) if max_radius > 0 else 0.3
            
            # Draw circle
            circle = Circle(
                (center.real, center.imag),
                radius,
                fill=False,
                edgecolor='blue',
                alpha=opacity,
                linewidth=1
            )
            self._ax.add_patch(circle)
        
        # Draw trace path (accumulated points)
        if len(self._trace_points) > 1:
            trace_real = [p.real for p in self._trace_points]
            trace_imag = [p.imag for p in self._trace_points]
            self._ax.plot(
                trace_real,
                trace_imag,
                'r-',
                linewidth=2,
                alpha=0.8,
                label='Trace path'
            )
        
        # Draw current drawing point (highlighted)
        self._ax.plot(
            state.trace_point.real,
            state.trace_point.imag,
            'ro',
            markersize=8,
            label='Current point'
        )
        
        # Display frame info (frame number and completion percentage)
        self._display_frame_info()
        
        # Apply view transformation (zoom/pan)
        self._apply_view_transform()
        
        # Set equal aspect ratio and grid
        self._ax.set_aspect('equal')
        self._ax.grid(True, alpha=0.3)
        self._ax.legend(loc='upper right', fontsize=8)
        
        # Update the display
        if self._fig is not None:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
    
    def _render_pyqtgraph(self, state: EpicycleState) -> None:
        """
        Render frame using PyQtGraph backend.
        
        Args:
            state: Current epicycle state to render
        
        Note:
            PyQtGraph implementation is a placeholder for future enhancement.
            Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            "PyQtGraph backend is not yet implemented. Use 'matplotlib' backend."
        )
    
    def _apply_view_transform(self) -> None:
        """
        Apply zoom and pan transformations to the view.
        
        Updates the axis limits based on current view center and scale.
        """
        if self._ax is None:
            return
        
        # Calculate view bounds based on center and scale
        # Larger scale = zoomed out, smaller scale = zoomed in
        view_width = 1000 / self._view_scale
        view_height = 1000 / self._view_scale
        
        self._ax.set_xlim(
            self._view_center.real - view_width / 2,
            self._view_center.real + view_width / 2
        )
        self._ax.set_ylim(
            self._view_center.imag - view_height / 2,
            self._view_center.imag + view_height / 2
        )
    
    def _display_frame_info(self) -> None:
        """
        Display current frame number and completion percentage.
        
        Shows real-time frame information in the top-left corner of the plot.
        """
        if self._ax is None:
            return
        
        # Calculate completion percentage
        if self._total_frames > 0:
            completion = (self._current_frame / self._total_frames) * 100
        else:
            completion = 0.0
        
        # Format frame info text
        info_text = f"Frame: {self._current_frame}/{self._total_frames}\n"
        info_text += f"Progress: {completion:.1f}%"
        
        # Display text in top-left corner (in axes coordinates)
        self._ax.text(
            0.02, 0.98,
            info_text,
            transform=self._ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    def animate(
        self,
        engine: EpicycleEngine,
        fps: int = 30,
        num_frames: Optional[int] = None,
        speed: float = 1.0
    ) -> None:
        """
        Run animation loop, maintaining target FPS.
        
        Creates a matplotlib animation that renders epicycle states
        at the specified frame rate. The animation continues until
        manually stopped or all frames are rendered.
        
        Args:
            engine: EpicycleEngine to generate animation frames
            fps: Target frames per second (default: 30)
            num_frames: Total number of frames (default: fps * 10 for 10 seconds)
            speed: Animation speed multiplier (default: 1.0)
        
        Raises:
            ValueError: If fps <= 0 or speed not in [0.1, 10.0]
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        
        if not (0.1 <= speed <= 10.0):
            raise ValueError(f"speed must be in [0.1, 10.0], got {speed}")
        
        # Default to 10 seconds of animation
        if num_frames is None:
            num_frames = fps * 10
        
        # Set total frames for progress tracking
        self._total_frames = num_frames
        self._current_frame = 0
        
        # Reset trace points for new animation
        self._trace_points = []
        
        # Setup matplotlib figure and axes
        if self.backend == "matplotlib":
            self._fig, self._ax = plt.subplots(figsize=(10, 10))
            self._fig.suptitle('Epicycle Animation', fontsize=14)
            
            # Generate all frames from the engine
            frames = list(engine.generate_animation_frames(num_frames, speed))
            
            # Create animation function
            def update_frame(frame_idx: int) -> None:
                """Update function called for each animation frame."""
                if frame_idx < len(frames):
                    state = frames[frame_idx]
                    self.render_frame(state, frame_idx)
            
            # Create matplotlib animation
            interval = 1000 / fps  # milliseconds per frame
            self._animation = FuncAnimation(
                self._fig,
                update_frame,
                frames=num_frames,
                interval=interval,
                repeat=True,
                blit=False
            )
            
            # Show the animation
            plt.show()
        
        elif self.backend == "pyqtgraph":
            raise NotImplementedError(
                "PyQtGraph backend is not yet implemented. Use 'matplotlib' backend."
            )
    
    def clear_trace(self) -> None:
        """
        Clear the accumulated trace path.
        
        Useful for resetting the visualization without recreating the renderer.
        """
        self._trace_points = []
    
    def set_view_center(self, center: complex) -> None:
        """
        Set the center point of the view (for panning).
        
        Args:
            center: Complex number representing the view center
        """
        self._view_center = center
    
    def set_view_scale(self, scale: float) -> None:
        """
        Set the view scale (for zooming).
        
        Args:
            scale: Scale factor (1.0 = normal, >1.0 = zoomed out, <1.0 = zoomed in)
        
        Raises:
            ValueError: If scale <= 0
        """
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        
        self._view_scale = scale
    
    def zoom_in(self, factor: float = 1.5) -> None:
        """
        Zoom in by the specified factor.
        
        Args:
            factor: Zoom factor (default: 1.5)
        """
        self._view_scale /= factor
    
    def zoom_out(self, factor: float = 1.5) -> None:
        """
        Zoom out by the specified factor.
        
        Args:
            factor: Zoom factor (default: 1.5)
        """
        self._view_scale *= factor
    
    def pan(self, delta_x: float, delta_y: float) -> None:
        """
        Pan the view by the specified delta.
        
        Args:
            delta_x: Horizontal pan distance
            delta_y: Vertical pan distance
        """
        self._view_center += complex(delta_x, delta_y)
