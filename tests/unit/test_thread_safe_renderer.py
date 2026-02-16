"""
Unit tests for ThreadSafeRenderer.

Tests thread-safe rendering operations and synchronization.
"""

import pytest
import threading
from unittest.mock import Mock, patch

from fourier_encryption.concurrency import ThreadSafeRenderer
from fourier_encryption.models.data_models import EpicycleState


class TestThreadSafeRenderer:
    """Test suite for ThreadSafeRenderer."""
    
    @pytest.fixture
    def renderer(self):
        """Create a ThreadSafeRenderer instance."""
        return ThreadSafeRenderer(backend="matplotlib")
    
    def test_initialization(self):
        """Test renderer initialization."""
        renderer = ThreadSafeRenderer(backend="matplotlib")
        
        assert renderer.backend == "matplotlib"
        assert renderer.renderer is not None
        assert renderer._render_lock is not None
    
    def test_attach_observer(self, renderer):
        """Test attaching an observer."""
        observer = Mock()
        
        renderer.attach_observer(observer)
        
        # Verify observer was attached to underlying renderer
        assert observer in renderer.renderer.observers
    
    def test_detach_observer(self, renderer):
        """Test detaching an observer."""
        observer = Mock()
        
        renderer.attach_observer(observer)
        renderer.detach_observer(observer)
        
        # Verify observer was removed
        assert observer not in renderer.renderer.observers
    
    def test_render_frame_thread_safe(self, renderer):
        """Test thread-safe frame rendering."""
        state = EpicycleState(
            time=0.0,
            positions=[complex(0, 0), complex(1, 1)],
            trace_point=complex(2, 2),
        )
        
        # Mock the underlying renderer to avoid matplotlib issues
        with patch.object(renderer.renderer, 'render_frame') as mock_render:
            renderer.render_frame(state, frame_number=0)
            
            # Verify underlying render was called
            mock_render.assert_called_once_with(state, 0)
    
    def test_render_frame_error_handling(self, renderer):
        """Test error handling during rendering."""
        state = EpicycleState(
            time=0.0,
            positions=[complex(0, 0)],
            trace_point=complex(1, 1),
        )
        
        # Make underlying renderer raise an exception
        with patch.object(renderer.renderer, 'render_frame', side_effect=Exception("Render error")):
            # Should not raise exception (error is caught and logged)
            renderer.render_frame(state, frame_number=0)
    
    def test_clear_trace(self, renderer):
        """Test clearing trace path."""
        with patch.object(renderer.renderer, 'clear_trace') as mock_clear:
            renderer.clear_trace()
            
            mock_clear.assert_called_once()
    
    def test_set_view_center(self, renderer):
        """Test setting view center."""
        center = complex(100, 200)
        
        with patch.object(renderer.renderer, 'set_view_center') as mock_set:
            renderer.set_view_center(center)
            
            mock_set.assert_called_once_with(center)
    
    def test_set_view_scale(self, renderer):
        """Test setting view scale."""
        scale = 2.0
        
        with patch.object(renderer.renderer, 'set_view_scale') as mock_set:
            renderer.set_view_scale(scale)
            
            mock_set.assert_called_once_with(scale)
    
    def test_zoom_in(self, renderer):
        """Test zoom in operation."""
        with patch.object(renderer.renderer, 'zoom_in') as mock_zoom:
            renderer.zoom_in(factor=1.5)
            
            mock_zoom.assert_called_once_with(1.5)
    
    def test_zoom_out(self, renderer):
        """Test zoom out operation."""
        with patch.object(renderer.renderer, 'zoom_out') as mock_zoom:
            renderer.zoom_out(factor=1.5)
            
            mock_zoom.assert_called_once_with(1.5)
    
    def test_pan(self, renderer):
        """Test pan operation."""
        with patch.object(renderer.renderer, 'pan') as mock_pan:
            renderer.pan(10.0, 20.0)
            
            mock_pan.assert_called_once_with(10.0, 20.0)
    
    def test_trace_points_property(self, renderer):
        """Test getting trace points."""
        # Add some trace points to underlying renderer
        renderer.renderer._trace_points = [complex(0, 0), complex(1, 1), complex(2, 2)]
        
        trace_points = renderer.trace_points
        
        assert len(trace_points) == 3
        assert trace_points[0] == complex(0, 0)
        assert trace_points[1] == complex(1, 1)
        assert trace_points[2] == complex(2, 2)
        
        # Verify it's a copy (not the original list)
        assert trace_points is not renderer.renderer._trace_points
    
    def test_concurrent_rendering(self, renderer):
        """Test concurrent rendering from multiple threads."""
        import math
        
        num_threads = 5
        num_frames_per_thread = 10
        errors = []
        
        def render_frames(thread_id: int):
            """Render multiple frames from a thread."""
            try:
                for i in range(num_frames_per_thread):
                    # Ensure time is within [0, 2π]
                    time_value = (float(i) / num_frames_per_thread) * 2 * math.pi
                    
                    state = EpicycleState(
                        time=time_value,
                        positions=[complex(thread_id, i)],
                        trace_point=complex(thread_id + 1, i + 1),
                    )
                    
                    with patch.object(renderer.renderer, 'render_frame'):
                        renderer.render_frame(state, frame_number=i)
            except Exception as e:
                errors.append(e)
        
        # Create and start threads
        threads = [
            threading.Thread(target=render_frames, args=(i,))
            for i in range(num_threads)
        ]
        
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0
    
    def test_context_manager(self, renderer):
        """Test context manager protocol."""
        with renderer as r:
            assert r is renderer
        
        # Trace should be cleared after exit
        assert len(renderer.renderer._trace_points) == 0
    
    def test_reentrant_lock(self, renderer):
        """Test that the lock is reentrant (allows nested calls)."""
        def nested_operation():
            """Perform nested rendering operations."""
            with renderer._render_lock:
                # First level lock acquisition
                with renderer._render_lock:
                    # Second level lock acquisition (should not deadlock)
                    pass
        
        # Should complete without deadlock
        nested_operation()
