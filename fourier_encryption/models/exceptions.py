"""
Exception hierarchy for Fourier-Based Image Encryption System.

This module defines a comprehensive exception hierarchy with context support
for rich error messages, following the design specified in the requirements.
"""

from typing import Any, Dict, Optional


class FourierEncryptionError(Exception):
    """
    Base exception for all Fourier encryption system errors.
    
    Supports context information for rich error messages and debugging.
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and optional context.
        
        Args:
            message: Human-readable error description
            context: Additional context information (e.g., file paths, values)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return formatted error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ImageProcessingError(FourierEncryptionError):
    """
    Exception raised for image loading and preprocessing failures.
    
    Examples:
        - Unsupported image format
        - Image file not found
        - Corrupted image data
        - Edge detection failures
        - Contour extraction failures
    """
    pass


class EncryptionError(FourierEncryptionError):
    """
    Exception raised for encryption and decryption failures.
    
    Examples:
        - Encryption operation failed
        - Key derivation failed
        - Invalid encryption parameters
    """
    pass


class DecryptionError(EncryptionError):
    """
    Exception raised specifically for decryption failures.
    
    Examples:
        - Wrong decryption key
        - Tampered encrypted data
        - HMAC verification failed
        - Corrupted ciphertext
    """
    pass


class SerializationError(FourierEncryptionError):
    """
    Exception raised for serialization and deserialization failures.
    
    Examples:
        - Failed to serialize coefficients
        - Corrupted serialized data
        - Schema validation failed
        - Missing required fields
    """
    pass


class AIModelError(FourierEncryptionError):
    """
    Exception raised for AI model loading and inference failures.
    
    Examples:
        - Model file not found
        - Model loading failed
        - Inference operation failed
        - GPU operation failed
        - Unsupported model format
    """
    pass


class ConfigurationError(FourierEncryptionError):
    """
    Exception raised for configuration validation failures.
    
    Examples:
        - Invalid configuration file
        - Missing required configuration sections
        - Configuration values out of valid range
        - Malformed configuration data
    """
    pass
