"""
Performance profiling utilities.

This module provides decorators and utilities for profiling critical
code paths and identifying performance bottlenecks.
"""

import time
import functools
import logging
from typing import Callable, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Track performance metrics for operations.
    
    Collects timing data, call counts, and other metrics
    for performance analysis.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self._metrics: Dict[str, Dict[str, Any]] = {}
    
    def record_timing(self, operation: str, duration: float) -> None:
        """
        Record timing for an operation.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        if operation not in self._metrics:
            self._metrics[operation] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "avg_time": 0.0
            }
        
        metrics = self._metrics[operation]
        metrics["count"] += 1
        metrics["total_time"] += duration
        metrics["min_time"] = min(metrics["min_time"], duration)
        metrics["max_time"] = max(metrics["max_time"], duration)
        metrics["avg_time"] = metrics["total_time"] / metrics["count"]
    
    def get_metrics(self, operation: str = None) -> Dict[str, Any]:
        """
        Get metrics for an operation or all operations.
        
        Args:
            operation: Specific operation name, or None for all
            
        Returns:
            Dictionary of metrics
        """
        if operation:
            return self._metrics.get(operation, {})
        return self._metrics.copy()
    
    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()
    
    def print_summary(self) -> None:
        """Print a summary of all metrics."""
        if not self._metrics:
            logger.info("No performance metrics recorded")
            return
        
        logger.info("=== Performance Metrics Summary ===")
        for operation, metrics in sorted(self._metrics.items()):
            logger.info(f"\n{operation}:")
            logger.info(f"  Calls: {metrics['count']}")
            logger.info(f"  Total time: {metrics['total_time']:.4f}s")
            logger.info(f"  Avg time: {metrics['avg_time']:.4f}s")
            logger.info(f"  Min time: {metrics['min_time']:.4f}s")
            logger.info(f"  Max time: {metrics['max_time']:.4f}s")


# Global metrics instance
_global_metrics = PerformanceMetrics()


def timed_operation(operation_name: str = None):
    """
    Decorator to time function execution.
    
    Usage:
        @timed_operation("my_function")
        def my_function():
            # ... code ...
    
    Args:
        operation_name: Name for the operation (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                _global_metrics.record_timing(op_name, duration)
                
                # Log slow operations (> 1 second)
                if duration > 1.0:
                    logger.warning(
                        f"Slow operation: {op_name} took {duration:.4f}s"
                    )
        
        return wrapper
    return decorator


@contextmanager
def timed_block(operation_name: str):
    """
    Context manager for timing code blocks.
    
    Usage:
        with timed_block("my_operation"):
            # ... code to time ...
    
    Args:
        operation_name: Name for the operation
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        _global_metrics.record_timing(operation_name, duration)


def get_performance_metrics(operation: str = None) -> Dict[str, Any]:
    """
    Get performance metrics.
    
    Args:
        operation: Specific operation name, or None for all
        
    Returns:
        Dictionary of metrics
    """
    return _global_metrics.get_metrics(operation)


def print_performance_summary() -> None:
    """Print performance metrics summary."""
    _global_metrics.print_summary()


def clear_performance_metrics() -> None:
    """Clear all performance metrics."""
    _global_metrics.clear()


class MemoryTracker:
    """
    Track memory usage for operations.
    
    Provides utilities for monitoring memory consumption
    during critical operations.
    """
    
    def __init__(self):
        """Initialize memory tracker."""
        self._peak_memory = 0
        self._current_memory = 0
    
    def update(self, memory_bytes: int) -> None:
        """
        Update memory usage.
        
        Args:
            memory_bytes: Current memory usage in bytes
        """
        self._current_memory = memory_bytes
        self._peak_memory = max(self._peak_memory, memory_bytes)
    
    def get_peak_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self._peak_memory / (1024 * 1024)
    
    def get_current_mb(self) -> float:
        """Get current memory usage in MB."""
        return self._current_memory / (1024 * 1024)
    
    def reset(self) -> None:
        """Reset memory tracking."""
        self._peak_memory = 0
        self._current_memory = 0


# Global memory tracker
_global_memory_tracker = MemoryTracker()


def get_memory_tracker() -> MemoryTracker:
    """Get the global memory tracker."""
    return _global_memory_tracker
