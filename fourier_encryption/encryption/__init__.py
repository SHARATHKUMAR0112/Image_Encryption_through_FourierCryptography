"""
Encryption module for Fourier-Based Image Encryption System.

This module provides secure encryption capabilities including:
- Key management with cryptographic best practices
- Abstract encryption strategy interface
- AES-256-GCM encryption implementation
"""

from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.encryption.base_encryptor import EncryptionStrategy
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor

__all__ = [
    "KeyManager",
    "EncryptionStrategy",
    "AES256Encryptor",
]
