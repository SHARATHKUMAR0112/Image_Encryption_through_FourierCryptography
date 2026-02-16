"""
Unit tests for visualization module.

Tests the LiveRenderer class and Observer pattern implementation.
"""

import math
import pytest
import numpy as np

from fourier_encryption.visualization.live_renderer import LiveRenderer, RenderObserver
from fourier_encryption.models.data_models import EpicycleState, FourierCoefficient
from fourier_encryption.core.epicycle_engine import EpicycleEngine


class TestObserver(RenderObserver):
    """Test observer for tracking frame notifications."""
    
    def __init__(self):
        self.frames_received = []
    
    def on_frame_rendered(self, state: EpicycleState, frame_number: int) -> None:
        """Record frame notifications."""
        self.frames_received.append((state, frame_number))


class TestLiveRenderer:
    """Test suite for LiveRenderer class."""
    
    def test_init_with_valid_backend(self):
        """Test initialization with valid backend."""
        renderer = LiveRenderer(backend="matplotlib")
        assert renderer.backend == "matplotlib"
        assert len(renderer.observers) == 0
        assert len(renderer._trace_points) == 0
    
    def test_init_with_invalid_backend(self):
        """Test initialization with invalid backend raises error."""
        with pytest.raises(ValueError) as exc_info:
            LiveRenderer(backend="invalid")
        
        assert "backend must be one of" in str(exc_info.value)
    
    def test_attach_observer(self):
        """Test attaching observers."""
        renderer = LiveRenderer()
        observer1 = TestObserver()
        observer2 = TestObserver()
        
        renderer.attach_observer(observer1)
        assert len(renderer.observers) == 1
        
        renderer.attach_observer(observer2)
        assert len(renderer.observers) == 2
        
        # Attaching same observer twice should not duplicate
        renderer.attach_observer(observer1)
        assert len(renderer.observers) == 2
    
    def test_detach_observer(self):
        """Test detaching observers."""
        renderer = LiveRenderer()
        observer = TestObserver()
        
        renderer.attach_observer(observer)
        assert len(renderer.observers) == 1
        
        renderer.detach_observer(observer)
        assert len(renderer.observers) == 0
    
    def test_notify_observers(self):
        """Test observer notification."""
        renderer = LiveRenderer()
        observer1 = TestObserver()
        observer2 = TestObserver()
        
        renderer.attach_observer(observer1)
        renderer.attach_observer(observer2)
        
        # Create test state
        state = EpicycleState(
            time=0.0,
            positions=[complex(0, 0), complex(10, 0)],
            trace_point=complex(10, 0)
        )
        
        renderer.notify_observers(state, 5)
        
        assert len(observer1.frames_received) == 1
        assert len(observer2.frames_received) == 1
        assert observer1.frames_received[0][1] == 5
        assert observer2.frames_received[0][1] == 5
    
    def test_render_frame_updates_trace(self):
        """Test that render_frame adds to trace points."""
        renderer = LiveRenderer()
        
        state1 = EpicycleState(
            time=0.0,
            positions=[complex(0, 0)],
            trace_point=complex(10, 0)
        )
        
        state2 = EpicycleState(
            time=0.5,
            positions=[complex(0, 0)],
            trace_point=complex(20, 10)
        )
        
        renderer.render_frame(state1, 0)
        assert len(renderer._trace_points) == 1
        
        renderer.render_frame(state2, 1)
        assert len(renderer._trace_points) == 2
    
    def test_clear_trace(self):
        """Test clearing trace points."""
        renderer = LiveRenderer()
        
        state = EpicycleState(
            time=0.0,
            positions=[complex(0, 0)],
            trace_point=complex(10, 0)
        )
        
        renderer.render_frame(state, 0)
        assert len(renderer._trace_points) == 1
        
        renderer.clear_trace()
        assert len(renderer._trace_points) == 0
    
    def test_zoom_in(self):
        """Test zoom in functionality."""
        renderer = LiveRenderer()
        initial_scale = renderer._view_scale
        
        renderer.zoom_in(factor=2.0)
        assert renderer._view_scale == initial_scale / 2.0
    
    def test_zoom_out(self):
        """Test zoom out functionality."""
        renderer = LiveRenderer()
        initial_scale = renderer._view_scale
        
        renderer.zoom_out(factor=2.0)
        assert renderer._view_scale == initial_scale * 2.0
    
    def test_set_view_scale(self):
        """Test setting view scale directly."""
        renderer = LiveRenderer()
        
        renderer.set_view_scale(2.5)
        assert renderer._view_scale == 2.5
    
    def test_set_view_scale_invalid(self):
        """Test setting invalid view scale raises error."""
        renderer = LiveRenderer()
        
        with pytest.raises(ValueError) as exc_info:
            renderer.set_view_scale(-1.0)
        
        assert "scale must be positive" in str(exc_info.value)
    
    def test_pan(self):
        """Test pan functionality."""
        renderer = LiveRenderer()
        initial_center = renderer._view_center
        
        renderer.pan(10.0, 20.0)
        assert renderer._view_center == initial_center + complex(10.0, 20.0)
    
    def test_set_view_center(self):
        """Test setting view center directly."""
        renderer = LiveRenderer()
        
        new_center = complex(100, 200)
        renderer.set_view_center(new_center)
        assert renderer._view_center == new_center
    
    def test_animate_invalid_fps(self):
        """Test animate with invalid FPS raises error."""
        renderer = LiveRenderer()
        
        # Create minimal engine
        coeffs = [
            FourierCoefficient(
                frequency=0,
                amplitude=10.0,
                phase=0.0,
                complex_value=10.0 + 0j
            )
        ]
        engine = EpicycleEngine(coeffs)
        
        with pytest.raises(ValueError) as exc_info:
            renderer.animate(engine, fps=0)
        
        assert "fps must be positive" in str(exc_info.value)
    
    def test_animate_invalid_speed(self):
        """Test animate with invalid speed raises error."""
        renderer = LiveRenderer()
        
        # Create minimal engine
        coeffs = [
            FourierCoefficient(
                frequency=0,
                amplitude=10.0,
                phase=0.0,
                complex_value=10.0 + 0j
            )
        ]
        engine = EpicycleEngine(coeffs)
        
        with pytest.raises(ValueError) as exc_info:
            renderer.animate(engine, fps=30, speed=0.05)
        
        assert "speed must be in [0.1, 10.0]" in str(exc_info.value)
    
    def test_frame_info_tracking(self):
        """Test that frame info is tracked correctly."""
        renderer = LiveRenderer()
        
        # Set total frames
        renderer._total_frames = 100
        
        # Render some frames
        state = EpicycleState(
            time=0.0,
            positions=[complex(0, 0)],
            trace_point=complex(10, 0)
        )
        
        renderer.render_frame(state, 25)
        assert renderer._current_frame == 25
        
        renderer.render_frame(state, 50)
        assert renderer._current_frame == 50
    
    def test_pyqtgraph_backend_not_implemented(self):
        """Test that PyQtGraph backend raises NotImplementedError."""
        renderer = LiveRenderer(backend="pyqtgraph")
        
        state = EpicycleState(
            time=0.0,
            positions=[complex(0, 0)],
            trace_point=complex(10, 0)
        )
        
        with pytest.raises(NotImplementedError):
            renderer.render_frame(state, 0)
    
    def test_fps_maintenance_minimum_30fps(self):
        """
        Test FPS maintenance (minimum 30 FPS).
        
        Requirements: 3.6.3, 3.12.3, 3.3.5
        
        This test verifies that the renderer can maintain at least 30 FPS
        by measuring the frame processing time. Since we're testing without
        actual display, we verify that the core rendering logic (observer
        notifications, trace updates) can handle 30+ FPS.
        """
        import time
        
        renderer = LiveRenderer()
        
        # Attach an observer to simulate real usage
        observer = TestObserver()
        renderer.attach_observer(observer)
        
        # Create test coefficients for animation
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
                complex_value=50.0 * np.exp(1j * math.pi / 4)
            ),
            FourierCoefficient(
                frequency=2,
                amplitude=25.0,
                phase=math.pi / 2,
                complex_value=25.0 * np.exp(1j * math.pi / 2)
            )
        ]
        engine = EpicycleEngine(coeffs)
        
        # Generate frames and measure rendering time
        num_test_frames = 100  # Test with 100 frames
        frames = list(engine.generate_animation_frames(num_test_frames, speed=1.0))
        
        start_time = time.time()
        for i, state in enumerate(frames):
            renderer.render_frame(state, i)
        end_time = time.time()
        
        # Calculate actual FPS
        elapsed_time = end_time - start_time
        actual_fps = num_test_frames / elapsed_time if elapsed_time > 0 else float('inf')
        
        # Verify FPS is at least 30
        # Since we're not doing actual rendering (no matplotlib display),
        # the core logic should easily handle 30+ FPS
        # We use a very conservative threshold to account for slow test environments
        min_acceptable_fps = 30.0
        assert actual_fps >= min_acceptable_fps, (
            f"FPS {actual_fps:.1f} is below minimum acceptable {min_acceptable_fps} FPS. "
            f"Rendered {num_test_frames} frames in {elapsed_time:.3f}s. "
            f"Core rendering logic should support at least 30 FPS."
        )
        
        # Verify all frames were processed
        assert len(observer.frames_received) == num_test_frames
        assert len(renderer._trace_points) == num_test_frames
    
    def test_zoom_and_pan_update_view_state(self):
        """
        Test that zoom and pan controls update view state correctly.
        
        Requirements: 3.6.3
        
        This test verifies that zoom and pan operations properly update
        the internal view state (center and scale).
        """
        renderer = LiveRenderer()
        
        # Test initial state
        assert renderer._view_center == complex(0, 0)
        assert renderer._view_scale == 1.0
        
        # Test zoom in updates scale
        renderer.zoom_in(factor=2.0)
        assert renderer._view_scale == 0.5
        
        # Test zoom out updates scale
        renderer.zoom_out(factor=2.0)
        assert renderer._view_scale == 1.0
        
        # Test pan updates center
        renderer.pan(100.0, 50.0)
        assert renderer._view_center == complex(100.0, 50.0)
        
        # Test multiple pan operations accumulate
        renderer.pan(20.0, 30.0)
        assert renderer._view_center == complex(120.0, 80.0)
        
        # Test set_view_center directly
        renderer.set_view_center(complex(200.0, 150.0))
        assert renderer._view_center == complex(200.0, 150.0)
        
        # Test set_view_scale directly
        renderer.set_view_scale(3.5)
        assert renderer._view_scale == 3.5
    
    def test_observer_pattern_notifications(self):
        """
        Test observer pattern notifications work correctly.
        
        Requirements: 3.12.3
        
        This test verifies that the observer pattern implementation
        correctly notifies all registered observers when frames are rendered.
        """
        renderer = LiveRenderer()
        
        # Create multiple observers
        observer1 = TestObserver()
        observer2 = TestObserver()
        observer3 = TestObserver()
        
        # Attach observers
        renderer.attach_observer(observer1)
        renderer.attach_observer(observer2)
        renderer.attach_observer(observer3)
        
        # Create test states
        states = [
            EpicycleState(
                time=0.0,
                positions=[complex(0, 0), complex(10, 0)],
                trace_point=complex(10, 0)
            ),
            EpicycleState(
                time=0.5,
                positions=[complex(0, 0), complex(10, 5)],
                trace_point=complex(10, 5)
            ),
            EpicycleState(
                time=1.0,
                positions=[complex(0, 0), complex(10, 10)],
                trace_point=complex(10, 10)
            )
        ]
        
        # Render frames
        for i, state in enumerate(states):
            renderer.render_frame(state, i)
        
        # Verify all observers received all notifications
        assert len(observer1.frames_received) == 3
        assert len(observer2.frames_received) == 3
        assert len(observer3.frames_received) == 3
        
        # Verify frame numbers are correct
        for i in range(3):
            assert observer1.frames_received[i][1] == i
            assert observer2.frames_received[i][1] == i
            assert observer3.frames_received[i][1] == i
        
        # Verify states are passed correctly
        for i in range(3):
            assert observer1.frames_received[i][0] == states[i]
            assert observer2.frames_received[i][0] == states[i]
            assert observer3.frames_received[i][0] == states[i]
        
        # Test detaching an observer
        renderer.detach_observer(observer2)
        
        # Render another frame
        state4 = EpicycleState(
            time=1.5,
            positions=[complex(0, 0), complex(5, 10)],
            trace_point=complex(5, 10)
        )
        renderer.render_frame(state4, 3)
        
        # Observer1 and observer3 should receive the new frame
        assert len(observer1.frames_received) == 4
        assert len(observer3.frames_received) == 4
        
        # Observer2 should not receive the new frame (still at 3)
        assert len(observer2.frames_received) == 3


class TestRenderObserver:
    """Test suite for RenderObserver abstract base class."""
    
    def test_observer_interface(self):
        """Test that observer implements required interface."""
        observer = TestObserver()
        
        state = EpicycleState(
            time=0.0,
            positions=[complex(0, 0)],
            trace_point=complex(10, 0)
        )
        
        observer.on_frame_rendered(state, 0)
        assert len(observer.frames_received) == 1



class TestMonitoringDashboard:
    """Test suite for MonitoringDashboard class."""
    
    def test_init(self):
        """Test dashboard initialization."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        
        dashboard = MonitoringDashboard()
        
        assert dashboard.metrics.current_coefficient_index == 0
        assert dashboard.metrics.active_radius == 0.0
        assert dashboard.metrics.progress_percentage == 0.0
        assert dashboard.metrics.fps == 0.0
        assert dashboard.metrics.processing_time == 0.0
        assert dashboard.metrics.memory_usage_mb == 0.0
        assert dashboard.metrics.encryption_status == "idle"
    
    def test_update_metrics_thread_safe(self):
        """Test thread-safe metric updates."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        
        dashboard = MonitoringDashboard()
        
        new_metrics = Metrics(
            current_coefficient_index=10,
            active_radius=5.5,
            progress_percentage=50.0,
            fps=30.0,
            processing_time=2.5,
            memory_usage_mb=100.0,
            encryption_status="encrypting"
        )
        
        dashboard.update_metrics(new_metrics)
        
        retrieved = dashboard.metrics
        assert retrieved.current_coefficient_index == 10
        assert retrieved.active_radius == 5.5
        assert retrieved.progress_percentage == 50.0
        assert retrieved.fps == 30.0
        assert retrieved.encryption_status == "encrypting"
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping background monitoring."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        import time
        
        dashboard = MonitoringDashboard(update_interval=0.05)
        
        # Start monitoring
        dashboard.start_monitoring()
        assert dashboard._running is True
        assert dashboard._update_thread is not None
        assert dashboard._update_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        dashboard.stop_monitoring()
        assert dashboard._running is False
    
    def test_monitoring_updates_system_metrics(self):
        """Test that monitoring thread updates system metrics."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        import time
        
        dashboard = MonitoringDashboard(update_interval=0.05)
        
        initial_memory = dashboard.metrics.memory_usage_mb
        initial_time = dashboard.metrics.processing_time
        
        dashboard.start_monitoring()
        time.sleep(0.2)
        
        # System metrics should be updated
        assert dashboard.metrics.memory_usage_mb > 0
        assert dashboard.metrics.processing_time > initial_time
        
        dashboard.stop_monitoring()
    
    def test_update_callback(self):
        """Test custom update callback."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        import time
        
        dashboard = MonitoringDashboard(update_interval=0.05)
        
        call_count = [0]
        
        def custom_callback():
            call_count[0] += 1
            return Metrics(
                current_coefficient_index=call_count[0],
                active_radius=1.0,
                progress_percentage=10.0,
                fps=30.0,
                processing_time=1.0,
                memory_usage_mb=50.0,
                encryption_status="encrypting"
            )
        
        dashboard.set_update_callback(custom_callback)
        dashboard.start_monitoring()
        
        time.sleep(0.2)
        
        # Callback should have been invoked multiple times
        assert call_count[0] > 0
        assert dashboard.metrics.current_coefficient_index > 0
        
        dashboard.stop_monitoring()
    
    def test_display_terminal_mode(self):
        """Test terminal display rendering."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        
        dashboard = MonitoringDashboard()
        
        metrics = Metrics(
            current_coefficient_index=25,
            active_radius=12.5,
            progress_percentage=75.0,
            fps=30.5,
            processing_time=5.25,
            memory_usage_mb=150.5,
            encryption_status="encrypting"
        )
        
        dashboard.update_metrics(metrics)
        output = dashboard.display(mode="terminal")
        
        # Verify key information is in output
        assert "ENCRYPTING" in output
        assert "75.0%" in output
        assert "25" in output
        assert "12.5" in output
        assert "30.5" in output
        assert "150.5" in output
    
    def test_display_gui_mode(self):
        """Test GUI display rendering."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        import json
        
        dashboard = MonitoringDashboard()
        
        metrics = Metrics(
            current_coefficient_index=25,
            active_radius=12.5,
            progress_percentage=75.0,
            fps=30.5,
            processing_time=5.25,
            memory_usage_mb=150.5,
            encryption_status="encrypting"
        )
        
        dashboard.update_metrics(metrics)
        output = dashboard.display(mode="gui")
        
        # Parse JSON output
        data = json.loads(output)
        assert data["status"] == "encrypting"
        assert data["progress"] == 75.0
        assert data["coefficient_index"] == 25
        assert data["active_radius"] == 12.5
        assert data["fps"] == 30.5
        assert data["memory_usage_mb"] == 150.5
    
    def test_display_invalid_mode(self):
        """Test display with invalid mode raises error."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        
        dashboard = MonitoringDashboard()
        
        with pytest.raises(ValueError) as exc_info:
            dashboard.display(mode="invalid")
        
        assert "Unknown display mode" in str(exc_info.value)
    
    def test_context_manager(self):
        """Test using dashboard as context manager."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        import time
        
        with MonitoringDashboard(update_interval=0.05) as dashboard:
            assert dashboard._running is True
            time.sleep(0.1)
        
        # Should be stopped after exiting context
        assert dashboard._running is False
    
    def test_concurrent_metric_updates(self):
        """Test concurrent updates from multiple threads."""
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        import threading
        import time
        
        dashboard = MonitoringDashboard()
        
        def update_worker(index):
            for i in range(10):
                metrics = Metrics(
                    current_coefficient_index=index * 10 + i,
                    active_radius=float(i),
                    progress_percentage=float(i * 10),
                    fps=30.0,
                    processing_time=1.0,
                    memory_usage_mb=50.0,
                    encryption_status="encrypting"
                )
                dashboard.update_metrics(metrics)
                time.sleep(0.01)
        
        # Start multiple threads updating metrics
        threads = [threading.Thread(target=update_worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        final_metrics = dashboard.metrics
        assert final_metrics.encryption_status == "encrypting"
    
    def test_updates_dont_block_main_thread(self):
        """
        Test that metric updates don't block the main thread.
        
        Requirements: 3.7.6
        
        This test verifies that background monitoring and metric updates
        execute without blocking the main thread, allowing the application
        to continue processing.
        """
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        import time
        
        dashboard = MonitoringDashboard(update_interval=0.05)
        
        # Start monitoring in background
        dashboard.start_monitoring()
        
        # Simulate main thread work while monitoring runs
        main_thread_work_count = 0
        start_time = time.time()
        
        # Do work for 0.3 seconds
        while time.time() - start_time < 0.3:
            main_thread_work_count += 1
            # Simulate some work
            _ = sum(range(1000))
            
            # Update metrics from main thread
            metrics = Metrics(
                current_coefficient_index=main_thread_work_count,
                active_radius=1.0,
                progress_percentage=10.0,
                fps=30.0,
                processing_time=1.0,
                memory_usage_mb=50.0,
                encryption_status="encrypting"
            )
            dashboard.update_metrics(metrics)
        
        # Main thread should have done significant work
        # If updates were blocking, this count would be very low
        assert main_thread_work_count > 100, (
            f"Main thread only completed {main_thread_work_count} iterations, "
            "suggesting updates may be blocking"
        )
        
        # Monitoring thread should still be running
        assert dashboard._running is True
        assert dashboard._update_thread.is_alive()
        
        dashboard.stop_monitoring()
    
    def test_all_required_metrics_tracked(self):
        """
        Test that all required metrics are tracked.
        
        Requirements: 3.7.1, 3.7.2, 3.7.3, 3.7.4, 3.7.5, 3.7.6
        
        This test verifies that the dashboard tracks all metrics specified
        in the acceptance criteria:
        - AC 3.7.1: Current coefficient index
        - AC 3.7.2: Radius magnitude for active epicycle
        - AC 3.7.3: Reconstruction progress percentage
        - AC 3.7.4: Encryption/decryption status
        - AC 3.7.5: Performance metrics (FPS, processing time, memory usage)
        """
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        
        dashboard = MonitoringDashboard()
        
        # Create metrics with all required fields
        test_metrics = Metrics(
            current_coefficient_index=42,      # AC 3.7.1
            active_radius=15.75,               # AC 3.7.2
            progress_percentage=67.5,          # AC 3.7.3
            encryption_status="decrypting",    # AC 3.7.4
            fps=29.8,                          # AC 3.7.5
            processing_time=12.34,             # AC 3.7.5
            memory_usage_mb=256.5              # AC 3.7.5
        )
        
        dashboard.update_metrics(test_metrics)
        retrieved = dashboard.metrics
        
        # Verify all required metrics are tracked
        assert retrieved.current_coefficient_index == 42, "AC 3.7.1: Coefficient index not tracked"
        assert retrieved.active_radius == 15.75, "AC 3.7.2: Active radius not tracked"
        assert retrieved.progress_percentage == 67.5, "AC 3.7.3: Progress percentage not tracked"
        assert retrieved.encryption_status == "decrypting", "AC 3.7.4: Encryption status not tracked"
        assert retrieved.fps == 29.8, "AC 3.7.5: FPS not tracked"
        assert retrieved.processing_time == 12.34, "AC 3.7.5: Processing time not tracked"
        assert retrieved.memory_usage_mb == 256.5, "AC 3.7.5: Memory usage not tracked"
    
    def test_thread_safe_read_during_updates(self):
        """
        Test thread-safe reading of metrics during concurrent updates.
        
        Requirements: 3.7.6
        
        This test verifies that reading metrics while they're being updated
        from multiple threads is safe and doesn't cause data corruption or
        race conditions.
        """
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        import threading
        import time
        
        dashboard = MonitoringDashboard()
        errors = []
        
        # Valid status values to cycle through
        valid_statuses = ["encrypting", "decrypting", "idle"]
        
        def writer_thread(thread_id):
            """Continuously update metrics."""
            try:
                for i in range(50):
                    metrics = Metrics(
                        current_coefficient_index=thread_id * 100 + i,
                        active_radius=float(i),
                        progress_percentage=min(100.0, float(i * 2)),
                        fps=30.0,
                        processing_time=1.0,
                        memory_usage_mb=50.0,
                        encryption_status=valid_statuses[thread_id % len(valid_statuses)]
                    )
                    dashboard.update_metrics(metrics)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Writer {thread_id}: {e}")
        
        def reader_thread(thread_id):
            """Continuously read metrics."""
            try:
                for _ in range(100):
                    metrics = dashboard.metrics
                    # Verify metrics are valid (not corrupted)
                    assert isinstance(metrics.current_coefficient_index, int)
                    assert isinstance(metrics.active_radius, float)
                    assert isinstance(metrics.progress_percentage, float)
                    assert isinstance(metrics.fps, float)
                    assert isinstance(metrics.encryption_status, str)
                    # Verify values are in valid ranges
                    assert metrics.current_coefficient_index >= 0
                    assert metrics.active_radius >= 0
                    assert 0 <= metrics.progress_percentage <= 100
                    assert metrics.fps >= 0
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Reader {thread_id}: {e}")
        
        # Start multiple writer and reader threads
        writers = [threading.Thread(target=writer_thread, args=(i,)) for i in range(3)]
        readers = [threading.Thread(target=reader_thread, args=(i,)) for i in range(3)]
        
        all_threads = writers + readers
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    def test_monitoring_thread_is_daemon(self):
        """
        Test that monitoring thread is a daemon thread.
        
        Requirements: 3.7.6
        
        This test verifies that the monitoring thread is configured as a
        daemon thread, ensuring it won't prevent the application from exiting.
        """
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        
        dashboard = MonitoringDashboard()
        dashboard.start_monitoring()
        
        # Verify thread is daemon
        assert dashboard._update_thread.daemon is True, (
            "Monitoring thread should be daemon to avoid blocking application exit"
        )
        
        dashboard.stop_monitoring()
    
    def test_metrics_update_in_realtime(self):
        """
        Test that metrics update in real-time during monitoring.
        
        Requirements: 3.7.6
        
        This test verifies that the monitoring dashboard updates metrics
        in real-time without requiring manual refresh.
        """
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        import time
        
        dashboard = MonitoringDashboard(update_interval=0.05)
        
        call_count = [0]
        
        def realtime_callback():
            """Callback that returns different metrics each time."""
            call_count[0] += 1
            return Metrics(
                current_coefficient_index=call_count[0],
                active_radius=float(call_count[0]),
                progress_percentage=min(100.0, call_count[0] * 10.0),
                fps=30.0,
                processing_time=call_count[0] * 0.1,
                memory_usage_mb=50.0 + call_count[0],
                encryption_status="encrypting"
            )
        
        dashboard.set_update_callback(realtime_callback)
        dashboard.start_monitoring()
        
        # Wait for several update cycles
        time.sleep(0.25)
        
        # Verify metrics have been updated multiple times
        assert call_count[0] >= 3, "Callback should be invoked multiple times for real-time updates"
        
        # Verify metrics reflect the updates
        metrics = dashboard.metrics
        assert metrics.current_coefficient_index >= 3
        assert metrics.active_radius >= 3.0
        
        dashboard.stop_monitoring()
    
    def test_monitoring_handles_callback_errors_gracefully(self):
        """
        Test that monitoring thread handles callback errors gracefully.
        
        Requirements: 3.7.6
        
        This test verifies that if the update callback raises an exception,
        the monitoring thread continues running and doesn't crash.
        """
        from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
        from fourier_encryption.models.data_models import Metrics
        import time
        
        dashboard = MonitoringDashboard(update_interval=0.05)
        
        call_count = [0]
        
        def failing_callback():
            """Callback that fails on first call but succeeds later."""
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Simulated callback error")
            return Metrics(
                current_coefficient_index=call_count[0],
                active_radius=1.0,
                progress_percentage=10.0,
                fps=30.0,
                processing_time=1.0,
                memory_usage_mb=50.0,
                encryption_status="encrypting"
            )
        
        dashboard.set_update_callback(failing_callback)
        dashboard.start_monitoring()
        
        # Wait for multiple update cycles
        time.sleep(0.25)
        
        # Monitoring thread should still be running despite the error
        assert dashboard._running is True
        assert dashboard._update_thread.is_alive()
        
        # Callback should have been called multiple times (including the failed one)
        assert call_count[0] > 1, "Monitoring should continue after callback error"
        
        dashboard.stop_monitoring()
