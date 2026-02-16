"""
Thread-safe wrapper for LiveRenderer.

This module provides thread-safe rendering capabilities for concurrent
visualization updates, ensuring proper locking and synchronization.
"""

import logging
import threading
from typing import Optional, List

from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.models.data_models import EpicycleState
from fourier_encryption.visualization.live_renderer import LiveRenderer, RenderObserver


logger = logging.getLogger(__name__)


class ThreadSafeRenderer:
    """
    Thread-safe wrapper for LiveRenderer.
    
    Provides synchronized access to rendering operations, ensuring that
    multiple threads can safely update the visualization without race
    conditions or corruption.
    
    Features:
    - Thread-safe frame rendering
    - Synchronized observer notifications
    - Proper locking for shared state
    - Safe concurrent access to trace points
    
    Attributes:
        renderer: Underlying LiveRenderer instance
        render_lock: Lock for synchronizing render operations
    """
    
    def __init__(self, backend: str = "matplotlib"):
        """
        Initialize the thread-safe renderer.
        
        Args:
            backend: Rendering backend ("matplotlib" or "pyqtgraph")
        """
        self.renderer = LiveRenderer(backend=backend)
        self._render_lock = threading.RLock()  # Reentrant lock for nested calls
        
        logger.info(
            "ThreadSafeRenderer initialized",
            extra={"backend": backend}
        )
    
    def attach_observer(self, observer: RenderObserver) -> None:
        """
        Register an observer for frame updates (thread-safe).
        
        Args:
            observer: RenderObserver instance to register
        """
        with self._render_lock:
            self.renderer.attach_observer(observer)
    
    def detach_observer(self, observer: RenderObserver) -> None:
        """
        Unregister an observer (thread-safe).
        
        Args:
            observer: RenderObserver instance to unregister
        """
        with self._render_lock:
            self.renderer.detach_observer(observer)
    
    def render_frame(self, state: EpicycleState, frame_number: int = 0) -> None:
        """
        Render a single frame (thread-safe).
        
        This method ensures that only one thread can render at a time,
        preventing race conditions in the visualization state.
        
        Args:
            state: Current epicycle state to render
            frame_number: Sequential frame number for observer notification
        """
        with self._render_lock:
            try:
                self.renderer.render_frame(state, frame_number)
            except Exception as e:
                logger.error(
                    "Error rendering frame",
                    extra={
                        "frame_number": frame_number,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                # Don't re-raise to avoid crashing the rendering thread
    
    def animate(
        self,
        engine: EpicycleEngine,
        fps: int = 30,
        num_frames: Optional[int] = None,
        speed: float = 1.0
    ) -> None:
        """
        Run animation loop (thread-safe).
        
        Args:
            engine: EpicycleEngine to generate animation frames
            fps: Target frames per second (default: 30)
            num_frames: Total number of frames (default: fps * 10 for 10 seconds)
            speed: Animation speed multiplier (default: 1.0)
        """
        with self._render_lock:
            try:
                self.renderer.animate(engine, fps, num_frames, speed)
            except Exception as e:
                logger.error(
                    "Error during animation",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                raise
    
    def clear_trace(self) -> None:
        """
        Clear the accumulated trace path (thread-safe).
        """
        with self._render_lock:
            self.renderer.clear_trace()
    
    def set_view_center(self, center: complex) -> None:
        """
        Set the center point of the view (thread-safe).
        
        Args:
            center: Complex number representing the view center
        """
        with self._render_lock:
            self.renderer.set_view_center(center)
    
    def set_view_scale(self, scale: float) -> None:
        """
        Set the view scale (thread-safe).
        
        Args:
            scale: Scale factor (1.0 = normal, >1.0 = zoomed out, <1.0 = zoomed in)
        """
        with self._render_lock:
            self.renderer.set_view_scale(scale)
    
    def zoom_in(self, factor: float = 1.5) -> None:
        """
        Zoom in by the specified factor (thread-safe).
        
        Args:
            factor: Zoom factor (default: 1.5)
        """
        with self._render_lock:
            self.renderer.zoom_in(factor)
    
    def zoom_out(self, factor: float = 1.5) -> None:
        """
        Zoom out by the specified factor (thread-safe).
        
        Args:
            factor: Zoom factor (default: 1.5)
        """
        with self._render_lock:
            self.renderer.zoom_out(factor)
    
    def pan(self, delta_x: float, delta_y: float) -> None:
        """
        Pan the view by the specified delta (thread-safe).
        
        Args:
            delta_x: Horizontal pan distance
            delta_y: Vertical pan distance
        """
        with self._render_lock:
            self.renderer.pan(delta_x, delta_y)
    
    @property
    def backend(self) -> str:
        """Get the rendering backend."""
        return self.renderer.backend
    
    @property
    def trace_points(self) -> List[complex]:
        """
        Get a copy of the trace points (thread-safe).
        
        Returns:
            List of accumulated trace points
        """
        with self._render_lock:
            return self.renderer._trace_points.copy()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear_trace()
        return False
