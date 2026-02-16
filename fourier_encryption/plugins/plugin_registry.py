"""
Plugin registry for managing and discovering plugins.

Provides centralized registries for:
- Encryption strategy plugins
- AI model plugins

Uses singleton pattern to ensure single registry instance per type.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

from fourier_encryption.models.exceptions import ConfigurationError
from .base_plugin import Plugin, EncryptionPlugin, AIModelPlugin, PluginMetadata

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Base registry for managing plugins.
    
    Provides:
    - Plugin registration and lookup
    - Plugin validation
    - Plugin lifecycle management
    """
    
    def __init__(self, plugin_type: str):
        """
        Initialize plugin registry.
        
        Args:
            plugin_type: Type of plugins this registry manages
        """
        self.plugin_type = plugin_type
        self._plugins: Dict[str, Plugin] = {}
        self._initialized_plugins: Dict[str, bool] = {}
    
    def register(self, plugin: Plugin) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
            
        Raises:
            ConfigurationError: If plugin is invalid or already registered
        """
        metadata = plugin.metadata
        
        # Validate plugin type
        if metadata.plugin_type != self.plugin_type:
            raise ConfigurationError(
                f"Plugin type mismatch: expected '{self.plugin_type}', "
                f"got '{metadata.plugin_type}'"
            )
        
        # Check for duplicate registration
        if metadata.name in self._plugins:
            raise ConfigurationError(
                f"Plugin '{metadata.name}' is already registered"
            )
        
        # Register plugin
        self._plugins[metadata.name] = plugin
        self._initialized_plugins[metadata.name] = False
        
        logger.info(
            f"Registered {self.plugin_type} plugin: {metadata.name} v{metadata.version}",
            extra={
                "plugin_name": metadata.name,
                "plugin_version": metadata.version,
                "plugin_type": self.plugin_type,
            }
        )
    
    def unregister(self, plugin_name: str) -> None:
        """
        Unregister a plugin and clean up its resources.
        
        Args:
            plugin_name: Name of plugin to unregister
            
        Raises:
            ConfigurationError: If plugin not found
        """
        if plugin_name not in self._plugins:
            raise ConfigurationError(f"Plugin '{plugin_name}' not found")
        
        # Clean up if initialized
        if self._initialized_plugins.get(plugin_name, False):
            self._plugins[plugin_name].cleanup()
        
        # Remove from registry
        del self._plugins[plugin_name]
        del self._initialized_plugins[plugin_name]
        
        logger.info(
            f"Unregistered {self.plugin_type} plugin: {plugin_name}",
            extra={"plugin_name": plugin_name, "plugin_type": self.plugin_type}
        )
    
    def get(self, plugin_name: str) -> Optional[Plugin]:
        """
        Get a registered plugin by name.
        
        Args:
            plugin_name: Name of plugin to retrieve
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[PluginMetadata]:
        """
        List all registered plugins.
        
        Returns:
            List of plugin metadata
        """
        return [plugin.metadata for plugin in self._plugins.values()]
    
    def initialize_plugin(self, plugin_name: str, config: Dict) -> None:
        """
        Initialize a registered plugin.
        
        Args:
            plugin_name: Name of plugin to initialize
            config: Plugin configuration
            
        Raises:
            ConfigurationError: If plugin not found or initialization fails
        """
        if plugin_name not in self._plugins:
            raise ConfigurationError(f"Plugin '{plugin_name}' not found")
        
        if self._initialized_plugins[plugin_name]:
            logger.warning(f"Plugin '{plugin_name}' already initialized")
            return
        
        try:
            self._plugins[plugin_name].initialize(config)
            self._initialized_plugins[plugin_name] = True
            logger.info(f"Initialized plugin: {plugin_name}")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize plugin '{plugin_name}': {e}"
            )
    
    def cleanup_all(self) -> None:
        """Clean up all registered plugins."""
        for plugin_name, plugin in self._plugins.items():
            if self._initialized_plugins.get(plugin_name, False):
                try:
                    plugin.cleanup()
                    logger.info(f"Cleaned up plugin: {plugin_name}")
                except Exception as e:
                    logger.error(f"Error cleaning up plugin '{plugin_name}': {e}")


class EncryptionPluginRegistry(PluginRegistry):
    """
    Registry for encryption strategy plugins.
    
    Manages custom encryption algorithms including:
    - Post-quantum encryption (Kyber, Dilithium)
    - Hardware-accelerated encryption
    - Custom encryption schemes
    """
    
    _instance: Optional["EncryptionPluginRegistry"] = None
    
    def __new__(cls):
        """Singleton pattern: ensure only one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize encryption plugin registry."""
        if not hasattr(self, "_initialized"):
            super().__init__(plugin_type="encryption")
            self._initialized = True
    
    def register(self, plugin: EncryptionPlugin) -> None:
        """
        Register an encryption plugin.
        
        Args:
            plugin: EncryptionPlugin instance
            
        Raises:
            ConfigurationError: If plugin is not an EncryptionPlugin
        """
        if not isinstance(plugin, EncryptionPlugin):
            raise ConfigurationError(
                f"Plugin must be an instance of EncryptionPlugin, "
                f"got {type(plugin).__name__}"
            )
        super().register(plugin)
    
    def get_encryptor(self, plugin_name: str) -> Optional[EncryptionPlugin]:
        """
        Get an encryption plugin by name.
        
        Args:
            plugin_name: Name of encryption plugin
            
        Returns:
            EncryptionPlugin instance or None if not found
        """
        plugin = self.get(plugin_name)
        return plugin if isinstance(plugin, EncryptionPlugin) else None


class AIModelPluginRegistry(PluginRegistry):
    """
    Registry for AI model plugins.
    
    Manages custom AI models for:
    - Edge detection
    - Coefficient optimization
    - Anomaly detection
    """
    
    _instance: Optional["AIModelPluginRegistry"] = None
    
    def __new__(cls):
        """Singleton pattern: ensure only one registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize AI model plugin registry."""
        if not hasattr(self, "_initialized"):
            super().__init__(plugin_type="ai_model")
            self._initialized = True
    
    def register(self, plugin: AIModelPlugin) -> None:
        """
        Register an AI model plugin.
        
        Args:
            plugin: AIModelPlugin instance
            
        Raises:
            ConfigurationError: If plugin is not an AIModelPlugin
        """
        if not isinstance(plugin, AIModelPlugin):
            raise ConfigurationError(
                f"Plugin must be an instance of AIModelPlugin, "
                f"got {type(plugin).__name__}"
            )
        super().register(plugin)
    
    def get_model(self, plugin_name: str) -> Optional[AIModelPlugin]:
        """
        Get an AI model plugin by name.
        
        Args:
            plugin_name: Name of AI model plugin
            
        Returns:
            AIModelPlugin instance or None if not found
        """
        plugin = self.get(plugin_name)
        return plugin if isinstance(plugin, AIModelPlugin) else None
    
    def list_by_model_type(self, model_type: str) -> List[AIModelPlugin]:
        """
        List all AI model plugins of a specific type.
        
        Args:
            model_type: Type of model ("edge_detector", "optimizer", "anomaly_detector")
            
        Returns:
            List of matching AI model plugins
        """
        return [
            plugin for plugin in self._plugins.values()
            if isinstance(plugin, AIModelPlugin) and plugin.model_type == model_type
        ]
