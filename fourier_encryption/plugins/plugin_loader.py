"""
Plugin loader for automatic plugin discovery and loading.

Provides:
- Automatic discovery of plugins from directories
- Dynamic loading of plugin modules
- Plugin validation and registration
"""

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import List, Optional, Type

from fourier_encryption.models.exceptions import ConfigurationError
from .base_plugin import Plugin, EncryptionPlugin, AIModelPlugin
from .plugin_registry import (
    PluginRegistry,
    EncryptionPluginRegistry,
    AIModelPluginRegistry,
)

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Automatic plugin discovery and loading.
    
    Discovers plugins from:
    - User plugin directory (~/.fourier_encryption/plugins/)
    - System plugin directory (/etc/fourier_encryption/plugins/)
    - Custom plugin directories
    
    Plugins must:
    - Be Python modules (.py files)
    - Contain classes that inherit from Plugin
    - Implement all required abstract methods
    """
    
    def __init__(self):
        """Initialize plugin loader."""
        self.encryption_registry = EncryptionPluginRegistry()
        self.ai_model_registry = AIModelPluginRegistry()
        self._loaded_modules: List[str] = []
    
    def discover_plugins(self, plugin_dir: Path) -> List[Type[Plugin]]:
        """
        Discover plugin classes in a directory.
        
        Args:
            plugin_dir: Directory to search for plugins
            
        Returns:
            List of discovered plugin classes
            
        Raises:
            ConfigurationError: If plugin directory is invalid
        """
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {plugin_dir}")
            return []
        
        if not plugin_dir.is_dir():
            raise ConfigurationError(f"Plugin path is not a directory: {plugin_dir}")
        
        discovered_plugins: List[Type[Plugin]] = []
        
        # Search for Python files
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue  # Skip private modules
            
            try:
                plugin_classes = self._load_plugin_module(plugin_file)
                discovered_plugins.extend(plugin_classes)
            except Exception as e:
                logger.error(
                    f"Failed to load plugin from {plugin_file}: {e}",
                    extra={"plugin_file": str(plugin_file), "error": str(e)}
                )
        
        logger.info(
            f"Discovered {len(discovered_plugins)} plugin(s) in {plugin_dir}",
            extra={"plugin_dir": str(plugin_dir), "count": len(discovered_plugins)}
        )
        
        return discovered_plugins
    
    def _load_plugin_module(self, plugin_file: Path) -> List[Type[Plugin]]:
        """
        Load a plugin module and extract plugin classes.
        
        Args:
            plugin_file: Path to plugin Python file
            
        Returns:
            List of plugin classes found in module
        """
        module_name = f"fourier_encryption.plugins.external.{plugin_file.stem}"
        
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if spec is None or spec.loader is None:
            raise ConfigurationError(f"Cannot load module from {plugin_file}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        self._loaded_modules.append(module_name)
        
        # Find plugin classes in module
        plugin_classes: List[Type[Plugin]] = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if class is a plugin (but not the base classes)
            if (issubclass(obj, Plugin) and 
                obj not in [Plugin, EncryptionPlugin, AIModelPlugin] and
                not inspect.isabstract(obj)):
                plugin_classes.append(obj)
                logger.debug(f"Found plugin class: {name} in {plugin_file.name}")
        
        return plugin_classes
    
    def load_and_register(
        self,
        plugin_dir: Path,
        auto_initialize: bool = False,
        config: Optional[dict] = None
    ) -> int:
        """
        Discover, load, and register plugins from a directory.
        
        Args:
            plugin_dir: Directory containing plugins
            auto_initialize: Whether to automatically initialize plugins
            config: Configuration for plugin initialization
            
        Returns:
            Number of plugins successfully registered
            
        Raises:
            ConfigurationError: If plugin loading fails
        """
        plugin_classes = self.discover_plugins(plugin_dir)
        registered_count = 0
        
        for plugin_class in plugin_classes:
            try:
                # Instantiate plugin
                plugin_instance = plugin_class()
                
                # Register based on type
                if isinstance(plugin_instance, EncryptionPlugin):
                    self.encryption_registry.register(plugin_instance)
                    registry = self.encryption_registry
                elif isinstance(plugin_instance, AIModelPlugin):
                    self.ai_model_registry.register(plugin_instance)
                    registry = self.ai_model_registry
                else:
                    logger.warning(
                        f"Unknown plugin type: {plugin_class.__name__}, skipping"
                    )
                    continue
                
                registered_count += 1
                
                # Auto-initialize if requested
                if auto_initialize:
                    plugin_config = config or {}
                    registry.initialize_plugin(
                        plugin_instance.metadata.name,
                        plugin_config
                    )
                
            except Exception as e:
                logger.error(
                    f"Failed to register plugin {plugin_class.__name__}: {e}",
                    extra={"plugin_class": plugin_class.__name__, "error": str(e)}
                )
        
        logger.info(
            f"Registered {registered_count} plugin(s) from {plugin_dir}",
            extra={"plugin_dir": str(plugin_dir), "count": registered_count}
        )
        
        return registered_count
    
    def load_from_standard_locations(
        self,
        auto_initialize: bool = False,
        config: Optional[dict] = None
    ) -> int:
        """
        Load plugins from standard locations.
        
        Standard locations:
        - ~/.fourier_encryption/plugins/ (user plugins)
        - /etc/fourier_encryption/plugins/ (system plugins, Linux/Mac)
        - %APPDATA%/fourier_encryption/plugins/ (system plugins, Windows)
        
        Args:
            auto_initialize: Whether to automatically initialize plugins
            config: Configuration for plugin initialization
            
        Returns:
            Total number of plugins registered
        """
        total_registered = 0
        
        # User plugin directory
        user_plugin_dir = Path.home() / ".fourier_encryption" / "plugins"
        if user_plugin_dir.exists():
            total_registered += self.load_and_register(
                user_plugin_dir,
                auto_initialize,
                config
            )
        
        # System plugin directory (platform-specific)
        if sys.platform.startswith("win"):
            # Windows: %APPDATA%
            import os
            appdata = os.getenv("APPDATA")
            if appdata:
                system_plugin_dir = Path(appdata) / "fourier_encryption" / "plugins"
                if system_plugin_dir.exists():
                    total_registered += self.load_and_register(
                        system_plugin_dir,
                        auto_initialize,
                        config
                    )
        else:
            # Linux/Mac: /etc
            system_plugin_dir = Path("/etc/fourier_encryption/plugins")
            if system_plugin_dir.exists():
                total_registered += self.load_and_register(
                    system_plugin_dir,
                    auto_initialize,
                    config
                )
        
        return total_registered
    
    def cleanup(self) -> None:
        """Clean up all loaded plugins."""
        self.encryption_registry.cleanup_all()
        self.ai_model_registry.cleanup_all()
        
        # Unload modules
        for module_name in self._loaded_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        self._loaded_modules.clear()
        logger.info("Plugin loader cleanup complete")
