"""
Base plugin interfaces for extensibility.

Defines abstract base classes for different plugin types:
- Plugin: Base interface for all plugins
- EncryptionPlugin: Interface for custom encryption strategies
- AIModelPlugin: Interface for custom AI models
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from fourier_encryption.models.data_models import EncryptedPayload


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    
    name: str
    version: str
    author: str
    description: str
    plugin_type: str  # "encryption", "ai_model"
    dependencies: Dict[str, str]  # Package name -> version requirement
    
    def __post_init__(self):
        """Validate metadata fields."""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.version:
            raise ValueError("Plugin version cannot be empty")
        if not self._is_valid_semver(self.version):
            raise ValueError(f"Invalid semantic version: {self.version}")
        if self.plugin_type not in ["encryption", "ai_model"]:
            raise ValueError(f"Invalid plugin type: {self.plugin_type}")
    
    @staticmethod
    def _is_valid_semver(version: str) -> bool:
        """Check if version follows semantic versioning (major.minor.patch)."""
        parts = version.split(".")
        if len(parts) != 3:
            return False
        return all(part.isdigit() for part in parts)


class Plugin(ABC):
    """
    Abstract base class for all plugins.
    
    All plugins must implement:
    - metadata property: Return plugin metadata
    - initialize method: Set up plugin resources
    - cleanup method: Release plugin resources
    """
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin-specific configuration dictionary
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up plugin resources.
        
        Called when plugin is unloaded or system shuts down.
        """
        pass


class EncryptionPlugin(Plugin):
    """
    Abstract base class for custom encryption strategy plugins.
    
    Encryption plugins must implement the EncryptionStrategy interface
    plus plugin lifecycle methods.
    
    This enables support for:
    - Post-quantum encryption algorithms (Kyber, Dilithium)
    - Custom encryption schemes
    - Hardware-accelerated encryption
    """
    
    @abstractmethod
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        """
        Encrypt data with the given key.
        
        Args:
            data: Plaintext data to encrypt
            key: Encryption key (derived from password)
            
        Returns:
            EncryptedPayload with ciphertext, IV, HMAC, and metadata
            
        Raises:
            EncryptionError: If encryption fails
        """
        pass
    
    @abstractmethod
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        """
        Decrypt encrypted payload with the given key.
        
        Args:
            payload: Encrypted payload with ciphertext, IV, HMAC
            key: Decryption key (derived from password)
            
        Returns:
            Decrypted plaintext data
            
        Raises:
            DecryptionError: If decryption fails or HMAC verification fails
        """
        pass
    
    @abstractmethod
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using KDF.
        
        Args:
            password: User-provided password
            salt: Cryptographic salt
            
        Returns:
            Derived key suitable for encryption
        """
        pass


class AIModelPlugin(Plugin):
    """
    Abstract base class for custom AI model plugins.
    
    AI model plugins enable integration of custom models for:
    - Edge detection
    - Coefficient optimization
    - Anomaly detection
    
    Models can be PyTorch, TensorFlow, ONNX, or custom implementations.
    """
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Return the type of AI model.
        
        Returns:
            One of: "edge_detector", "optimizer", "anomaly_detector"
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """
        Load the AI model from disk.
        
        Args:
            model_path: Path to model file
            
        Raises:
            AIModelError: If model loading fails
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input array (shape depends on model type)
            
        Returns:
            Model predictions
            
        Raises:
            InferenceError: If inference fails
        """
        pass
    
    @abstractmethod
    def get_input_shape(self) -> tuple:
        """
        Return expected input shape for the model.
        
        Returns:
            Tuple representing input dimensions
        """
        pass
    
    @abstractmethod
    def get_output_shape(self) -> tuple:
        """
        Return expected output shape for the model.
        
        Returns:
            Tuple representing output dimensions
        """
        pass
    
    @property
    @abstractmethod
    def supports_gpu(self) -> bool:
        """
        Check if model supports GPU acceleration.
        
        Returns:
            True if GPU is supported, False otherwise
        """
        pass
