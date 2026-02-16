"""
Key management utilities for secure encryption operations.

This module provides cryptographically secure key generation, validation,
and comparison operations with timing-attack prevention.
"""

import secrets
import hmac
from typing import Optional


class KeyManager:
    """
    Secure key handling with cryptographic best practices.
    
    Provides:
    - Cryptographically secure random salt generation
    - Password strength validation
    - Constant-time key comparison (timing-attack prevention)
    """
    
    # Minimum password requirements
    MIN_PASSWORD_LENGTH = 12
    SALT_LENGTH = 32  # 256 bits
    
    def generate_salt(self) -> bytes:
        """
        Generate cryptographically secure random salt.
        
        Uses secrets module which provides cryptographically strong random
        numbers suitable for managing data such as passwords, account
        authentication, security tokens, and related secrets.
        
        Returns:
            32 bytes of cryptographically secure random data
            
        Example:
            >>> km = KeyManager()
            >>> salt = km.generate_salt()
            >>> len(salt)
            32
        """
        return secrets.token_bytes(self.SALT_LENGTH)
    
    def validate_key_strength(self, password: str) -> bool:
        """
        Validate password meets minimum strength requirements.
        
        Checks:
        - Minimum length (12 characters)
        - Contains at least one uppercase letter
        - Contains at least one lowercase letter
        - Contains at least one digit
        - Contains at least one special character
        
        Args:
            password: Password string to validate
            
        Returns:
            True if password meets all requirements, False otherwise
            
        Example:
            >>> km = KeyManager()
            >>> km.validate_key_strength("weak")
            False
            >>> km.validate_key_strength("StrongP@ssw0rd123")
            True
        """
        if not password or not isinstance(password, str):
            return False
        
        # Check minimum length
        if len(password) < self.MIN_PASSWORD_LENGTH:
            return False
        
        # Check for character diversity
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def constant_time_compare(self, a: bytes, b: bytes) -> bool:
        """
        Compare two byte sequences in constant time.
        
        Prevents timing attacks by ensuring comparison takes the same amount
        of time regardless of where the first difference occurs. Uses HMAC
        comparison which is designed to be constant-time.
        
        Args:
            a: First byte sequence
            b: Second byte sequence
            
        Returns:
            True if sequences are equal, False otherwise
            
        Security:
            This function is critical for preventing timing attacks during
            HMAC verification and key comparison. Never use == for comparing
            sensitive data like keys or HMACs.
            
        Example:
            >>> km = KeyManager()
            >>> km.constant_time_compare(b"secret", b"secret")
            True
            >>> km.constant_time_compare(b"secret", b"public")
            False
        """
        if not isinstance(a, bytes) or not isinstance(b, bytes):
            raise TypeError("Both arguments must be bytes")
        
        # hmac.compare_digest is designed for constant-time comparison
        return hmac.compare_digest(a, b)
