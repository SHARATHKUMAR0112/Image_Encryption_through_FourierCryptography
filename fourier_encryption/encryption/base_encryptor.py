"""
Abstract base class for encryption strategies.

This module defines the interface for encryption implementations,
following the Strategy pattern for algorithm flexibility.
"""

from abc import ABC, abstractmethod
from fourier_encryption.models.data_models import EncryptedPayload


class EncryptionStrategy(ABC):
    """
    Abstract base class for encryption strategies.
    
    Defines the interface that all encryption implementations must follow.
    This allows for flexible encryption algorithm selection (AES-256,
    post-quantum algorithms, etc.) while maintaining a consistent interface.
    
    Design Pattern: Strategy Pattern
    - Encapsulates encryption algorithms
    - Makes algorithms interchangeable
    - Allows runtime algorithm selection
    
    Requirements:
        - 3.12.1: Use Factory pattern for encryption strategy selection
        - 3.12.4: Use Abstract Base Classes for extensibility
    """
    
    @abstractmethod
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        """
        Encrypt data using the implemented encryption algorithm.
        
        Args:
            data: Plaintext data to encrypt (serialized coefficients)
            key: Encryption key (derived from password)
            
        Returns:
            EncryptedPayload containing ciphertext, IV, HMAC, and metadata
            
        Raises:
            EncryptionError: If encryption operation fails
            
        Security:
            - Must generate unique IV for each encryption
            - Must compute HMAC for integrity protection
            - Must securely wipe sensitive data after use
        """
        pass
    
    @abstractmethod
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        """
        Decrypt data using the implemented encryption algorithm.
        
        Args:
            payload: EncryptedPayload containing ciphertext, IV, and HMAC
            key: Decryption key (derived from password)
            
        Returns:
            Decrypted plaintext data (serialized coefficients)
            
        Raises:
            DecryptionError: If decryption fails or HMAC verification fails
            
        Security:
            - Must verify HMAC before decryption (constant-time)
            - Must fail gracefully with wrong key
            - Must detect tampered data
            - Must securely wipe sensitive data after use
        """
        pass
