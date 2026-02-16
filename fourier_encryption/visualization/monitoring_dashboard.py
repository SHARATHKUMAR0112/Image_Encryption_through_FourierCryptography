"""
Monitoring Dashboard for real-time metrics display.

This module provides a thread-safe monitoring dashboard that tracks and displays
performance metrics during encryption/decryption operations.
"""

import threading
import time
import psutil
from typing import Optional, Callable
from dataclasses import replace

from fourier_encryption.models.data_models import Metrics


class MonitoringDashboard:
    """
    Real-time metrics display dashboard.
    
    Provides thread-safe metric updates and display capabilities for monitoring
    encryption/decryption operations. Supports both terminal and GUI rendering.
    
    Attributes:
        metrics: Current metrics snapshot
        update_thread: Background thread for metric collection
        running: Flag indicating if monitoring is active
        update_callback: Optional callback for custom metric updates
    """
    
    def __init__(self, update_interval: float = 0.1):
        """
        Initialize the monitoring dashboard.
        
        Args:
            update_interval: Time in seconds between metric updates (default: 0.1)
        """
        self._metrics = Metrics(
            current_coefficient_index=0,
            active_radius=0.0,
            progress_percentage=0.0,
            fps=0.0,
            processing_time=0.0,
            memory_usage_mb=0.0,
            encryption_status="idle"
        )
        self._lock = threading.Lock()
        self._update_thread: Optional[threading.Thread] = None
        self._running = False
        self._update_interval = update_interval
        self._update_callback: Optional[Callable[[], Metrics]] = None
        self._start_time: Optional[float] = None
        self._process = psutil.Process()
        
    @property
    def metrics(self) -> Metrics:
        """
        Get current metrics snapshot (thread-safe).
        
        Returns:
            Current metrics
        """
        with self._lock:
            return self._metrics
    
    def update_metrics(self, metrics: Metrics) -> None:
        """
        Update metrics in a thread-safe manner.
        
        This method can be called from any thread to update the dashboard metrics.
        
        Args:
            metrics: New metrics to display
        """
        with self._lock:
            self._metrics = metrics
    
    def set_update_callback(self, callback: Callable[[], Metrics]) -> None:
        """
        Set a callback function for automatic metric updates.
        
        The callback will be invoked periodically by the background monitoring thread
        to fetch updated metrics.
        
        Args:
            callback: Function that returns updated Metrics
        """
        self._update_callback = callback
    
    def start_monitoring(self) -> None:
        """
        Start background metric collection thread.
        
        Launches a daemon thread that periodically collects system metrics
        (memory usage, processing time) and invokes the update callback if set.
        The thread does not block the main thread.
        """
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        self._update_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MonitoringDashboard"
        )
        self._update_thread.start()
    
    def stop_monitoring(self) -> None:
        """
        Stop background metric collection thread.
        
        Signals the monitoring thread to stop and waits for it to complete.
        """
        if not self._running:
            return
        
        self._running = False
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
    
    def _monitoring_loop(self) -> None:
        """
        Background monitoring loop.
        
        Runs in a separate thread, periodically collecting system metrics
        and invoking the update callback.
        """
        while self._running:
            try:
                # Collect system metrics
                memory_mb = self._process.memory_info().rss / (1024 * 1024)
                processing_time = time.time() - self._start_time if self._start_time else 0.0
                
                # Get updated metrics from callback or update system metrics
                if self._update_callback:
                    try:
                        updated_metrics = self._update_callback()
                        with self._lock:
                            self._metrics = updated_metrics
                    except Exception:
                        # Silently ignore callback errors to avoid crashing the thread
                        pass
                else:
                    # Update only system metrics
                    with self._lock:
                        self._metrics = replace(
                            self._metrics,
                            memory_usage_mb=memory_mb,
                            processing_time=processing_time
                        )
                
                time.sleep(self._update_interval)
            except Exception:
                # Silently ignore errors to keep monitoring thread alive
                time.sleep(self._update_interval)
    
    def display(self, mode: str = "terminal") -> str:
        """
        Render dashboard display.
        
        Generates a formatted display of current metrics. Supports both terminal
        and GUI rendering modes.
        
        Args:
            mode: Display mode - "terminal" for text output, "gui" for structured data
        
        Returns:
            Formatted display string (for terminal mode) or structured data (for GUI mode)
        """
        metrics = self.metrics
        
        if mode == "terminal":
            return self._render_terminal(metrics)
        elif mode == "gui":
            return self._render_gui_data(metrics)
        else:
            raise ValueError(f"Unknown display mode: {mode}")
    
    def _render_terminal(self, metrics: Metrics) -> str:
        """
        Render metrics for terminal display.
        
        Args:
            metrics: Metrics to display
        
        Returns:
            Formatted terminal output string
        """
        lines = [
            "=" * 60,
            "  FOURIER ENCRYPTION MONITORING DASHBOARD",
            "=" * 60,
            "",
            f"Status:              {metrics.encryption_status.upper()}",
            f"Progress:            {metrics.progress_percentage:.1f}%",
            "",
            "--- Processing Metrics ---",
            f"Coefficient Index:   {metrics.current_coefficient_index}",
            f"Active Radius:       {metrics.active_radius:.4f}",
            f"Processing Time:     {metrics.processing_time:.2f}s",
            "",
            "--- Performance Metrics ---",
            f"FPS:                 {metrics.fps:.1f}",
            f"Memory Usage:        {metrics.memory_usage_mb:.1f} MB",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def _render_gui_data(self, metrics: Metrics) -> str:
        """
        Render metrics as structured data for GUI display.
        
        Args:
            metrics: Metrics to display
        
        Returns:
            JSON-formatted string with metric data
        """
        import json
        
        data = {
            "status": metrics.encryption_status,
            "progress": metrics.progress_percentage,
            "coefficient_index": metrics.current_coefficient_index,
            "active_radius": metrics.active_radius,
            "processing_time": metrics.processing_time,
            "fps": metrics.fps,
            "memory_usage_mb": metrics.memory_usage_mb
        }
        return json.dumps(data, indent=2)
    
    def __enter__(self):
        """Context manager entry - start monitoring."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop monitoring."""
        self.stop_monitoring()
        return False
