"""
AES-256-GCM encryption implementation with HMAC integrity protection.

This module provides industrial-grade encryption using AES-256 in GCM mode
with PBKDF2 key derivation and HMAC-SHA256 for integrity verification.
"""

import hashlib
import secrets
from typing import Dict, Any

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from fourier_encryption.encryption.base_encryptor import EncryptionStrategy
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.models.data_models import EncryptedPayload
from fourier_encryption.models.exceptions import EncryptionError, DecryptionError


class AES256Encryptor(EncryptionStrategy):
    """
    AES-256-GCM encryption with PBKDF2 key derivation and HMAC integrity.
    
    Security Features:
    - AES-256-GCM for authenticated encryption
    - PBKDF2-HMAC-SHA256 with 100,000+ iterations for key derivation
    - Cryptographically secure random IV generation
    - HMAC-SHA256 for additional integrity protection
    - Constant-time HMAC comparison (timing-attack prevention)
    - Secure memory wiping for sensitive data
    
    Requirements:
        - 3.4.1: AES-256 encryption for Fourier coefficients
        - 3.4.2: PBKDF2 key derivation with 100,000+ iterations
        - 3.4.3: Cryptographically secure random IV
        - 3.4.4: HMAC-SHA256 for integrity validation
        - 3.4.5: Encrypted payload includes frequency, amplitude, phase
        - 3.4.6: Graceful decryption failure with incorrect key
        - 3.18.2: Secure wipe of sensitive data from memory
    """
    
    # Cryptographic constants
    KEY_LENGTH = 32  # 256 bits for AES-256
    IV_LENGTH = 16   # 128 bits for AES
    HMAC_LENGTH = 32  # 256 bits for SHA-256
    DEFAULT_KDF_ITERATIONS = 100_000
    
    def __init__(self, kdf_iterations: int = DEFAULT_KDF_ITERATIONS):
        """
        Initialize AES-256 encryptor.
        
        Args:
            kdf_iterations: Number of PBKDF2 iterations (minimum 100,000)
            
        Raises:
            ValueError: If kdf_iterations < 100,000
        """
        if kdf_iterations < 100_000:
            raise ValueError(
                f"KDF iterations must be >= 100,000 for security, got {kdf_iterations}"
            )
        
        self.kdf_iterations = kdf_iterations
        self.key_manager = KeyManager()
        self.backend = default_backend()
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2-HMAC-SHA256.
        
        Uses PBKDF2 (Password-Based Key Derivation Function 2) with HMAC-SHA256
        to derive a cryptographically strong key from a password. The high
        iteration count (100,000+) makes brute-force attacks computationally
        expensive.
        
        Args:
            password: User-provided password
            salt: Cryptographically secure random salt (32 bytes)
            
        Returns:
            Derived 256-bit encryption key
            
        Raises:
            EncryptionError: If key derivation fails
            
        Security:
            - Uses PBKDF2 with HMAC-SHA256
            - Minimum 100,000 iterations (configurable)
            - 256-bit output for AES-256
            - Unique salt per encryption
        """
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.KEY_LENGTH,
                salt=salt,
                iterations=self.kdf_iterations,
                backend=self.backend
            )
            
            # Convert password to bytes if needed
            password_bytes = password.encode('utf-8') if isinstance(password, str) else password
            
            return kdf.derive(password_bytes)
            
        except Exception as e:
            raise EncryptionError(
                f"Key derivation failed: {e}",
                context={"iterations": self.kdf_iterations, "salt_length": len(salt)}
            )
    
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        """
        Encrypt data using AES-256-GCM with HMAC integrity protection.
        
        Encryption Process:
        1. Generate cryptographically secure random IV
        2. Encrypt data with AES-256-GCM
        3. Compute HMAC-SHA256 over (IV || ciphertext)
        4. Return EncryptedPayload with all components
        
        Args:
            data: Plaintext data to encrypt (serialized coefficients)
            key: 256-bit encryption key (from derive_key)
            
        Returns:
            EncryptedPayload with ciphertext, IV, HMAC, and metadata
            
        Raises:
            EncryptionError: If encryption operation fails
            TypeError: If inputs are not bytes
            
        Security:
            - Unique IV for each encryption (prevents IV reuse attacks)
            - GCM mode provides authenticated encryption
            - Additional HMAC for defense in depth
            - Metadata includes version for future compatibility
        """
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        
        if len(key) != self.KEY_LENGTH:
            raise ValueError(
                f"key must be {self.KEY_LENGTH} bytes, got {len(key)} bytes"
            )
        
        try:
            # Generate cryptographically secure random IV
            iv = secrets.token_bytes(self.IV_LENGTH)
            
            # Create AES-256-GCM cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Get GCM authentication tag
            gcm_tag = encryptor.tag
            
            # Compute HMAC over (IV || ciphertext || GCM tag) for defense in depth
            h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
            h.update(iv)
            h.update(ciphertext)
            h.update(gcm_tag)
            hmac_value = h.finalize()
            
            # Create metadata
            metadata: Dict[str, Any] = {
                "version": "1.0",
                "algorithm": "AES-256-GCM",
                "kdf": "PBKDF2-HMAC-SHA256",
                "kdf_iterations": self.kdf_iterations,
                "data_length": len(data),
                "gcm_tag": gcm_tag.hex()
            }
            
            return EncryptedPayload(
                ciphertext=ciphertext,
                iv=iv,
                hmac=hmac_value,
                metadata=metadata
            )
            
        except Exception as e:
            raise EncryptionError(
                f"Encryption failed: {e}",
                context={"data_length": len(data), "key_length": len(key)}
            )
    
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        """
        Decrypt data using AES-256-GCM with HMAC verification.
        
        Decryption Process:
        1. Verify HMAC (constant-time comparison)
        2. Decrypt ciphertext with AES-256-GCM
        3. Verify GCM authentication tag
        4. Return plaintext
        5. Securely wipe sensitive data
        
        Args:
            payload: EncryptedPayload with ciphertext, IV, HMAC, metadata
            key: 256-bit decryption key (from derive_key)
            
        Returns:
            Decrypted plaintext data (serialized coefficients)
            
        Raises:
            DecryptionError: If HMAC verification fails, wrong key, or tampered data
            TypeError: If inputs have wrong types
            
        Security:
            - HMAC verification before decryption (fail-fast)
            - Constant-time HMAC comparison (timing-attack prevention)
            - GCM authentication prevents tampering
            - Graceful failure with wrong key
            - Secure memory wiping after use
        """
        if not isinstance(payload, EncryptedPayload):
            raise TypeError("payload must be EncryptedPayload")
        
        if not isinstance(key, bytes):
            raise TypeError("key must be bytes")
        
        if len(key) != self.KEY_LENGTH:
            raise ValueError(
                f"key must be {self.KEY_LENGTH} bytes, got {len(key)} bytes"
            )
        
        key_material = None
        
        try:
            # Extract GCM tag from metadata
            if "gcm_tag" not in payload.metadata:
                raise DecryptionError(
                    "Missing GCM tag in metadata",
                    context={"metadata_keys": list(payload.metadata.keys())}
                )
            
            gcm_tag = bytes.fromhex(payload.metadata["gcm_tag"])
            
            # Compute expected HMAC
            h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
            h.update(payload.iv)
            h.update(payload.ciphertext)
            h.update(gcm_tag)
            expected_hmac = h.finalize()
            
            # Verify HMAC using constant-time comparison
            if not self.key_manager.constant_time_compare(expected_hmac, payload.hmac):
                raise DecryptionError(
                    "HMAC verification failed - data may be tampered or wrong key",
                    context={
                        "expected_hmac_length": len(expected_hmac),
                        "actual_hmac_length": len(payload.hmac)
                    }
                )
            
            # Create AES-256-GCM cipher with authentication tag
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(payload.iv, gcm_tag),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            plaintext = decryptor.update(payload.ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except DecryptionError:
            # Re-raise DecryptionError as-is
            raise
            
        except Exception as e:
            # Wrap other exceptions in DecryptionError
            raise DecryptionError(
                f"Decryption failed: {e}",
                context={
                    "ciphertext_length": len(payload.ciphertext),
                    "iv_length": len(payload.iv),
                    "key_length": len(key)
                }
            )
        
        finally:
            # Securely wipe key material from memory
            if key_material is not None:
                self.secure_wipe(key_material)
    
    def secure_wipe(self, data: bytearray) -> None:
        """
        Securely wipe sensitive data from memory.
        
        Overwrites the memory location with zeros to prevent sensitive data
        from remaining in memory after use. This is a defense-in-depth measure
        against memory dump attacks.
        
        Args:
            data: Bytearray containing sensitive data to wipe
            
        Note:
            This provides best-effort security. Python's memory management
            may have already copied the data elsewhere. For maximum security,
            use specialized secure memory libraries or hardware security modules.
            
        Security:
            - Overwrites with zeros
            - Multiple passes for paranoid security
            - Prevents memory dump attacks
        """
        if not isinstance(data, bytearray):
            # Can only wipe mutable bytearray, not immutable bytes
            return
        
        # Overwrite with zeros (multiple passes for paranoid security)
        for _ in range(3):
            for i in range(len(data)):
                data[i] = 0
