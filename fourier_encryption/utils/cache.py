"""
Caching utilities for performance optimization.

This module provides caching mechanisms for frequently accessed data
to reduce redundant computations and improve overall system performance.
"""

from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple
import hashlib
import pickle


class DataCache:
    """
    Generic data cache with size limits and TTL support.
    
    Provides a simple key-value cache for storing frequently accessed
    data such as preprocessed images, computed coefficients, or
    configuration objects.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize cache with maximum size.
        
        Args:
            max_size: Maximum number of items to store in cache
        """
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._access_order = []  # Track access order for LRU eviction
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key in self._cache:
            # Update access order (move to end = most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Store item in cache.
        
        If cache is full, evicts least recently used item.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # If key already exists, update it
        if key in self._cache:
            self._cache[key] = value
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return
        
        # If cache is full, evict LRU item
        if len(self._cache) >= self._max_size:
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        
        # Add new item
        self._cache[key] = value
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class ComputationCache:
    """
    Cache for expensive computation results.
    
    Automatically generates cache keys from function arguments
    and caches the results of expensive operations.
    """
    
    def __init__(self, max_size: int = 50):
        """
        Initialize computation cache.
        
        Args:
            max_size: Maximum number of results to cache
        """
        self._cache = DataCache(max_size=max_size)
    
    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """
        Generate cache key from function name and arguments.
        
        Args:
            func_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Hash-based cache key
        """
        # Create a hashable representation of arguments
        try:
            # Try to pickle arguments for hashing
            args_bytes = pickle.dumps((args, sorted(kwargs.items())))
            args_hash = hashlib.sha256(args_bytes).hexdigest()[:16]
        except (pickle.PicklingError, TypeError):
            # Fallback to string representation
            args_str = f"{args}_{sorted(kwargs.items())}"
            args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
        
        return f"{func_name}_{args_hash}"
    
    def get_or_compute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Get cached result or compute and cache it.
        
        Args:
            func: Function to call if result not cached
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Cached or computed result
        """
        key = self._generate_key(func.__name__, *args, **kwargs)
        
        # Check cache
        result = self._cache.get(key)
        if result is not None:
            return result
        
        # Compute and cache
        result = func(*args, **kwargs)
        self._cache.put(key, result)
        
        return result
    
    def clear(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()


# Global computation cache instance
_global_computation_cache = ComputationCache(max_size=100)


def cached_computation(func: Callable) -> Callable:
    """
    Decorator for caching expensive computation results.
    
    Usage:
        @cached_computation
        def expensive_function(x, y):
            # ... expensive computation ...
            return result
    
    Args:
        func: Function to cache
        
    Returns:
        Wrapped function with caching
    """
    def wrapper(*args, **kwargs):
        return _global_computation_cache.get_or_compute(func, *args, **kwargs)
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def clear_computation_cache() -> None:
    """Clear the global computation cache."""
    _global_computation_cache.clear()


# Specialized caches for specific data types
class ImageCache(DataCache):
    """Cache for preprocessed images."""
    
    def __init__(self):
        super().__init__(max_size=20)  # Images are large, keep fewer


class CoefficientCache(DataCache):
    """Cache for computed Fourier coefficients."""
    
    def __init__(self):
        super().__init__(max_size=50)  # Coefficients are smaller


class ConfigCache(DataCache):
    """Cache for loaded configuration objects."""
    
    def __init__(self):
        super().__init__(max_size=10)  # Few configs needed


# Global cache instances
image_cache = ImageCache()
coefficient_cache = CoefficientCache()
config_cache = ConfigCache()
