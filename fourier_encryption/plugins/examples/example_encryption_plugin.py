"""
Example encryption plugin demonstrating the plugin interface.

This is a simple XOR-based encryption for demonstration purposes only.
DO NOT USE IN PRODUCTION - this is not cryptographically secure!
"""

import hashlib
import hmac
import os
from typing import Dict, Any

from fourier_encryption.plugins.base_plugin import EncryptionPlugin, PluginMetadata
from fourier_encryption.models.data_models import EncryptedPayload
from fourier_encryption.models.exceptions import EncryptionError, DecryptionError


class ExampleXOREncryptor(EncryptionPlugin):
    """
    Example XOR-based encryption plugin.
    
    WARNING: This is for demonstration only! XOR encryption is NOT secure
    for real-world use. Use AES-256 or post-quantum algorithms in production.
    
    This plugin demonstrates:
    - Plugin metadata definition
    - Initialization and cleanup
    - Key derivation
    - Encryption and decryption
    - HMAC integrity protection
    """
    
    def __init__(self):
        """Initialize the plugin."""
        self._initialized = False
        self.kdf_iterations = 10000
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="example-xor-encryptor",
            version="1.0.0",
            author="Fourier Encryption Team",
            description="Example XOR encryption plugin (demonstration only)",
            plugin_type="encryption",
            dependencies={}  # No external dependencies
        )
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Configuration dictionary
                - kdf_iterations: Number of PBKDF2 iterations (default: 10000)
        """
        self.kdf_iterations = config.get("kdf_iterations", 10000)
        self._initialized = True
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self._initialized = False
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: User-provided password
            salt: Cryptographic salt (32 bytes)
            
        Returns:
            Derived key (32 bytes)
        """
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            self.kdf_iterations,
            dklen=32
        )
    
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        """
        Encrypt data using XOR with key stream.
        
        Args:
            data: Plaintext data to encrypt
            key: Encryption key (32 bytes)
            
        Returns:
            EncryptedPayload with ciphertext, IV, HMAC, and metadata
            
        Raises:
            EncryptionError: If encryption fails
        """
        if not self._initialized:
            raise EncryptionError("Plugin not initialized")
        
        try:
            # Generate random IV
            iv = os.urandom(16)
            
            # Generate key stream from key and IV
            key_stream = self._generate_key_stream(key, iv, len(data))
            
            # XOR data with key stream
            ciphertext = bytes(a ^ b for a, b in zip(data, key_stream))
            
            # Compute HMAC over IV + ciphertext
            hmac_value = hmac.new(
                key,
                iv + ciphertext,
                hashlib.sha256
            ).digest()
            
            return EncryptedPayload(
                ciphertext=ciphertext,
                iv=iv,
                hmac=hmac_value,
                metadata={
                    "algorithm": "XOR-Stream",
                    "version": "1.0",
                    "kdf_iterations": self.kdf_iterations,
                }
            )
            
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}")
    
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        """
        Decrypt encrypted payload.
        
        Args:
            payload: Encrypted payload with ciphertext, IV, HMAC
            key: Decryption key (32 bytes)
            
        Returns:
            Decrypted plaintext data
            
        Raises:
            DecryptionError: If decryption or HMAC verification fails
        """
        if not self._initialized:
            raise DecryptionError("Plugin not initialized")
        
        try:
            # Verify HMAC first (constant-time comparison)
            expected_hmac = hmac.new(
                key,
                payload.iv + payload.ciphertext,
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(expected_hmac, payload.hmac):
                raise DecryptionError("HMAC verification failed - data may be tampered")
            
            # Generate same key stream
            key_stream = self._generate_key_stream(key, payload.iv, len(payload.ciphertext))
            
            # XOR ciphertext with key stream to recover plaintext
            plaintext = bytes(a ^ b for a, b in zip(payload.ciphertext, key_stream))
            
            return plaintext
            
        except DecryptionError:
            raise
        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}")
    
    def _generate_key_stream(self, key: bytes, iv: bytes, length: int) -> bytes:
        """
        Generate pseudo-random key stream from key and IV.
        
        Uses SHA-256 in counter mode to generate key stream.
        
        Args:
            key: Encryption key
            iv: Initialization vector
            length: Length of key stream to generate
            
        Returns:
            Key stream bytes
        """
        key_stream = bytearray()
        counter = 0
        
        while len(key_stream) < length:
            # Hash key + IV + counter
            h = hashlib.sha256()
            h.update(key)
            h.update(iv)
            h.update(counter.to_bytes(4, 'big'))
            key_stream.extend(h.digest())
            counter += 1
        
        return bytes(key_stream[:length])
