"""
Utility modules for performance optimization and caching.
"""

from fourier_encryption.utils.cache import (
    DataCache,
    ComputationCache,
    cached_computation,
    clear_computation_cache,
    image_cache,
    coefficient_cache,
    config_cache,
)

from fourier_encryption.utils.profiling import (
    PerformanceMetrics,
    timed_operation,
    timed_block,
    get_performance_metrics,
    print_performance_summary,
    clear_performance_metrics,
    MemoryTracker,
    get_memory_tracker,
)

__all__ = [
    # Cache utilities
    "DataCache",
    "ComputationCache",
    "cached_computation",
    "clear_computation_cache",
    "image_cache",
    "coefficient_cache",
    "config_cache",
    # Profiling utilities
    "PerformanceMetrics",
    "timed_operation",
    "timed_block",
    "get_performance_metrics",
    "print_performance_summary",
    "clear_performance_metrics",
    "MemoryTracker",
    "get_memory_tracker",
]
