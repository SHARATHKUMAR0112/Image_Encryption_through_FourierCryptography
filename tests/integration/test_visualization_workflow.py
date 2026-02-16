"""
Integration tests for visualization workflow.

Tests the complete visualization pipeline with epicycle engine.
"""

import math
import pytest
import numpy as np

from fourier_encryption.visualization.live_renderer import LiveRenderer, RenderObserver
from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.core.epicycle_engine import EpicycleEngine


class MetricsObserver(RenderObserver):
    """Observer that tracks metrics during animation."""
    
    def __init__(self):
        self.frame_count = 0
        self.trace_lengths = []
    
    def on_frame_rendered(self, state, frame_number):
        """Track frame metrics."""
        self.frame_count += 1
        # Note: We can't access _trace_points directly, but we can track state
        self.trace_lengths.append(frame_number)


class TestVisualizationWorkflow:
    """Integration tests for complete visualization workflow."""
    
    def test_simple_circle_animation(self):
        """Test animating a simple circle with one coefficient."""
        # Create a single coefficient representing a circle
        coeffs = [
            FourierCoefficient(
                frequency=1,
                amplitude=50.0,
                phase=0.0,
                complex_value=50.0 + 0j
            )
        ]
        
        # Create engine and renderer
        engine = EpicycleEngine(coeffs)
        renderer = LiveRenderer(backend="matplotlib")
        
        # Attach observer
        observer = MetricsObserver()
        renderer.attach_observer(observer)
        
        # Manually render a few frames (not full animation)
        num_frames = 10
        for i, state in enumerate(engine.generate_animation_frames(num_frames)):
            renderer.render_frame(state, i)
        
        # Verify observer was notified
        assert observer.frame_count == num_frames
        assert len(renderer._trace_points) == num_frames
    
    def test_multi_epicycle_animation(self):
        """Test animating multiple epicycles."""
        # Create multiple coefficients
        coeffs = [
            FourierCoefficient(
                frequency=0,
                amplitude=100.0,
                phase=0.0,
                complex_value=100.0 + 0j
            ),
            FourierCoefficient(
                frequency=1,
                amplitude=50.0,
                phase=math.pi / 4,
                complex_value=50.0 * complex(math.cos(math.pi/4), math.sin(math.pi/4))
            ),
            FourierCoefficient(
                frequency=2,
                amplitude=25.0,
                phase=-math.pi / 3,
                complex_value=25.0 * complex(math.cos(-math.pi/3), math.sin(-math.pi/3))
            )
        ]
        
        # Create engine and renderer
        engine = EpicycleEngine(coeffs)
        renderer = LiveRenderer(backend="matplotlib")
        
        # Render frames
        num_frames = 20
        for i, state in enumerate(engine.generate_animation_frames(num_frames)):
            renderer.render_frame(state, i)
            
            # Verify state has correct number of positions
            assert len(state.positions) == len(coeffs)
        
        # Verify trace was accumulated
        assert len(renderer._trace_points) == num_frames
    
    def test_zoom_and_pan_during_animation(self):
        """Test zoom and pan controls work during animation."""
        coeffs = [
            FourierCoefficient(
                frequency=1,
                amplitude=50.0,
                phase=0.0,
                complex_value=50.0 + 0j
            )
        ]
        
        engine = EpicycleEngine(coeffs)
        renderer = LiveRenderer(backend="matplotlib")
        
        # Initial view state
        initial_scale = renderer._view_scale
        initial_center = renderer._view_center
        
        # Zoom in
        renderer.zoom_in(2.0)
        assert renderer._view_scale == initial_scale / 2.0
        
        # Pan
        renderer.pan(100, 50)
        assert renderer._view_center == initial_center + complex(100, 50)
        
        # Render a frame with modified view
        state = engine.compute_state(0.0)
        renderer.render_frame(state, 0)
        
        # View transformations should persist
        assert renderer._view_scale == initial_scale / 2.0
        assert renderer._view_center == initial_center + complex(100, 50)
    
    def test_frame_info_updates(self):
        """Test that frame info is updated correctly during animation."""
        coeffs = [
            FourierCoefficient(
                frequency=1,
                amplitude=50.0,
                phase=0.0,
                complex_value=50.0 + 0j
            )
        ]
        
        engine = EpicycleEngine(coeffs)
        renderer = LiveRenderer(backend="matplotlib")
        
        # Set total frames
        total_frames = 100
        renderer._total_frames = total_frames
        
        # Render frames at different points
        test_frames = [0, 25, 50, 75, 99]
        
        for frame_num in test_frames:
            t = (frame_num / total_frames) * 2 * math.pi
            state = engine.compute_state(t)
            renderer.render_frame(state, frame_num)
            
            # Verify current frame is tracked
            assert renderer._current_frame == frame_num
    
    def test_clear_trace_between_animations(self):
        """Test clearing trace between multiple animations."""
        coeffs = [
            FourierCoefficient(
                frequency=1,
                amplitude=50.0,
                phase=0.0,
                complex_value=50.0 + 0j
            )
        ]
        
        engine = EpicycleEngine(coeffs)
        renderer = LiveRenderer(backend="matplotlib")
        
        # First animation
        for i, state in enumerate(engine.generate_animation_frames(10)):
            renderer.render_frame(state, i)
        
        assert len(renderer._trace_points) == 10
        
        # Clear trace
        renderer.clear_trace()
        assert len(renderer._trace_points) == 0
        
        # Second animation
        for i, state in enumerate(engine.generate_animation_frames(5)):
            renderer.render_frame(state, i)
        
        assert len(renderer._trace_points) == 5
    
    def test_multiple_observers(self):
        """Test multiple observers receive notifications."""
        coeffs = [
            FourierCoefficient(
                frequency=1,
                amplitude=50.0,
                phase=0.0,
                complex_value=50.0 + 0j
            )
        ]
        
        engine = EpicycleEngine(coeffs)
        renderer = LiveRenderer(backend="matplotlib")
        
        # Attach multiple observers
        observer1 = MetricsObserver()
        observer2 = MetricsObserver()
        
        renderer.attach_observer(observer1)
        renderer.attach_observer(observer2)
        
        # Render frames
        num_frames = 15
        for i, state in enumerate(engine.generate_animation_frames(num_frames)):
            renderer.render_frame(state, i)
        
        # Both observers should receive all notifications
        assert observer1.frame_count == num_frames
        assert observer2.frame_count == num_frames
    
    def test_animation_with_different_speeds(self):
        """Test animation with different speed settings."""
        coeffs = [
            FourierCoefficient(
                frequency=1,
                amplitude=50.0,
                phase=0.0,
                complex_value=50.0 + 0j
            )
        ]
        
        engine = EpicycleEngine(coeffs)
        renderer = LiveRenderer(backend="matplotlib")
        
        # Test different speeds
        speeds = [0.5, 1.0, 2.0]
        
        for speed in speeds:
            renderer.clear_trace()
            
            num_frames = 20
            frames = list(engine.generate_animation_frames(num_frames, speed))
            
            # Verify we get the expected number of frames
            assert len(frames) == num_frames
            
            # Render all frames
            for i, state in enumerate(frames):
                renderer.render_frame(state, i)
            
            assert len(renderer._trace_points) == num_frames
    
    def test_observer_detachment(self):
        """Test that detached observers don't receive notifications."""
        coeffs = [
            FourierCoefficient(
                frequency=1,
                amplitude=50.0,
                phase=0.0,
                complex_value=50.0 + 0j
            )
        ]
        
        engine = EpicycleEngine(coeffs)
        renderer = LiveRenderer(backend="matplotlib")
        
        observer = MetricsObserver()
        renderer.attach_observer(observer)
        
        # Render some frames
        for i, state in enumerate(engine.generate_animation_frames(5)):
            renderer.render_frame(state, i)
        
        assert observer.frame_count == 5
        
        # Detach observer
        renderer.detach_observer(observer)
        
        # Render more frames
        for i, state in enumerate(engine.generate_animation_frames(5)):
            renderer.render_frame(state, i + 5)
        
        # Observer should not have received new notifications
        assert observer.frame_count == 5
