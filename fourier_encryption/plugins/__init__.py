"""
Plugin system for extensibility.

This module provides a plugin architecture for:
- Custom encryption strategies
- Custom AI models
- Future post-quantum encryption algorithms

The plugin system uses a registry pattern with automatic discovery.
"""

from .plugin_registry import (
    PluginRegistry,
    EncryptionPluginRegistry,
    AIModelPluginRegistry,
)
from .base_plugin import (
    Plugin,
    EncryptionPlugin,
    AIModelPlugin,
)
from .plugin_loader import PluginLoader

__all__ = [
    "PluginRegistry",
    "EncryptionPluginRegistry",
    "AIModelPluginRegistry",
    "Plugin",
    "EncryptionPlugin",
    "AIModelPlugin",
    "PluginLoader",
]
