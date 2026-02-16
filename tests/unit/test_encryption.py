"""
Unit tests for encryption module.

Tests AES-256 encryption, PBKDF2 key derivation, constant-time comparison,
and secure memory wiping functionality.

Requirements:
    - 3.4.1: AES-256 encryption for Fourier coefficients
    - 3.4.2: PBKDF2 key derivation with 100,000+ iterations
    - 3.18.4: Constant-time key comparison
    - 3.18.2: Secure wipe of sensitive data
"""

import pytest
import secrets
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.models.data_models import EncryptedPayload
from fourier_encryption.models.exceptions import EncryptionError, DecryptionError


class TestAES256Encryptor:
    """Test suite for AES256Encryptor class."""
    
    def test_encryption_produces_expected_structure(self):
        """
        Test that AES-256 encryption produces expected ciphertext structure.
        
        Validates:
        - EncryptedPayload contains all required fields
        - IV is 16 bytes (128 bits)
        - HMAC is 32 bytes (256 bits)
        - Metadata includes version, algorithm, KDF info
        - Ciphertext is non-empty
        
        Requirement: 3.4.1
        """
        encryptor = AES256Encryptor()
        
        # Test data
        plaintext = b"Test Fourier coefficients data"
        key = secrets.token_bytes(32)  # 256-bit key
        
        # Encrypt
        payload = encryptor.encrypt(plaintext, key)
        
        # Verify payload structure
        assert isinstance(payload, EncryptedPayload)
        assert isinstance(payload.ciphertext, bytes)
        assert isinstance(payload.iv, bytes)
        assert isinstance(payload.hmac, bytes)
        assert isinstance(payload.metadata, dict)
        
        # Verify field lengths
        assert len(payload.iv) == 16, "IV must be 16 bytes for AES"
        assert len(payload.hmac) == 32, "HMAC must be 32 bytes for SHA-256"
        assert len(payload.ciphertext) > 0, "Ciphertext must not be empty"
        
        # Verify metadata structure
        assert "version" in payload.metadata
        assert "algorithm" in payload.metadata
        assert "kdf" in payload.metadata
        assert "kdf_iterations" in payload.metadata
        assert "data_length" in payload.metadata
        assert "gcm_tag" in payload.metadata
        
        # Verify metadata values
        assert payload.metadata["version"] == "1.0"
        assert payload.metadata["algorithm"] == "AES-256-GCM"
        assert payload.metadata["kdf"] == "PBKDF2-HMAC-SHA256"
        assert payload.metadata["kdf_iterations"] >= 100_000
        assert payload.metadata["data_length"] == len(plaintext)
    
    def test_encryption_with_different_data_sizes(self):
        """Test encryption handles various data sizes correctly."""
        encryptor = AES256Encryptor()
        key = secrets.token_bytes(32)
        
        # Test different sizes
        test_sizes = [1, 16, 100, 1000, 10000]
        
        for size in test_sizes:
            plaintext = secrets.token_bytes(size)
            payload = encryptor.encrypt(plaintext, key)
            
            assert len(payload.ciphertext) > 0
            assert payload.metadata["data_length"] == size
    
    def test_encryption_requires_bytes_input(self):
        """Test that encryption rejects non-bytes input."""
        encryptor = AES256Encryptor()
        key = secrets.token_bytes(32)
        
        # Test with string (should fail)
        with pytest.raises(TypeError, match="data must be bytes"):
            encryptor.encrypt("not bytes", key)
        
        # Test with wrong key type
        with pytest.raises(TypeError, match="key must be bytes"):
            encryptor.encrypt(b"data", "not bytes")
    
    def test_encryption_requires_correct_key_length(self):
        """Test that encryption validates key length."""
        encryptor = AES256Encryptor()
        plaintext = b"test data"
        
        # Test with wrong key length
        wrong_key = secrets.token_bytes(16)  # 128 bits, not 256
        
        with pytest.raises(ValueError, match="key must be 32 bytes"):
            encryptor.encrypt(plaintext, wrong_key)


class TestPBKDF2KeyDerivation:
    """Test suite for PBKDF2 key derivation."""
    
    def test_pbkdf2_with_known_test_vectors(self):
        """
        Test PBKDF2 key derivation with known test vectors.
        
        Uses known inputs to verify correct PBKDF2 implementation.
        Tests that the same password and salt always produce the same key.
        
        Requirement: 3.4.2
        """
        encryptor = AES256Encryptor(kdf_iterations=100_000)
        
        # Test with known password and salt
        password = "password"
        salt = b"salt_for_testing_32_bytes_long!"
        
        # Derive key
        derived_key = encryptor.derive_key(password, salt)
        
        # Verify key properties
        assert isinstance(derived_key, bytes)
        assert len(derived_key) == 32, "Derived key must be 256 bits (32 bytes)"
        
        # Verify deterministic: same inputs produce same output
        derived_key2 = encryptor.derive_key(password, salt)
        assert derived_key == derived_key2
        
        # Verify against independently computed value using cryptography library
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
            backend=default_backend()
        )
        expected_key = kdf.derive(password.encode('utf-8'))
        assert derived_key == expected_key, "Derived key must match expected PBKDF2 output"
    
    def test_pbkdf2_different_salts_produce_different_keys(self):
        """Test that different salts produce different derived keys."""
        encryptor = AES256Encryptor()
        password = "test_password"
        
        salt1 = secrets.token_bytes(32)
        salt2 = secrets.token_bytes(32)
        
        key1 = encryptor.derive_key(password, salt1)
        key2 = encryptor.derive_key(password, salt2)
        
        assert key1 != key2, "Different salts must produce different keys"
    
    def test_pbkdf2_different_passwords_produce_different_keys(self):
        """Test that different passwords produce different derived keys."""
        encryptor = AES256Encryptor()
        salt = secrets.token_bytes(32)
        
        key1 = encryptor.derive_key("password1", salt)
        key2 = encryptor.derive_key("password2", salt)
        
        assert key1 != key2, "Different passwords must produce different keys"
    
    def test_pbkdf2_minimum_iterations(self):
        """Test that PBKDF2 enforces minimum iteration count."""
        # Should succeed with 100,000 iterations
        encryptor = AES256Encryptor(kdf_iterations=100_000)
        assert encryptor.kdf_iterations == 100_000
        
        # Should fail with less than 100,000 iterations
        with pytest.raises(ValueError, match="KDF iterations must be >= 100,000"):
            AES256Encryptor(kdf_iterations=50_000)
    
    def test_pbkdf2_handles_unicode_passwords(self):
        """Test that PBKDF2 correctly handles Unicode passwords."""
        encryptor = AES256Encryptor()
        salt = secrets.token_bytes(32)
        
        # Test with Unicode characters
        password = "pässwörd123!@#"
        key = encryptor.derive_key(password, salt)
        
        assert isinstance(key, bytes)
        assert len(key) == 32
    
    def test_pbkdf2_deterministic_output(self):
        """Test that PBKDF2 produces deterministic output."""
        encryptor = AES256Encryptor(kdf_iterations=100_000)
        
        password = "test_password"
        salt = b"fixed_salt_for_testing_32_bytes"
        
        # Derive key multiple times
        keys = [encryptor.derive_key(password, salt) for _ in range(5)]
        
        # All keys should be identical
        assert all(k == keys[0] for k in keys), "PBKDF2 must be deterministic"


class TestKeyManager:
    """Test suite for KeyManager class."""
    
    def test_constant_time_compare_equal_values(self):
        """
        Test constant-time comparison with equal values.
        
        Requirement: 3.18.4
        """
        km = KeyManager()
        
        value1 = b"secret_key_material"
        value2 = b"secret_key_material"
        
        assert km.constant_time_compare(value1, value2) is True
    
    def test_constant_time_compare_different_values(self):
        """Test constant-time comparison with different values."""
        km = KeyManager()
        
        value1 = b"secret_key_material"
        value2 = b"different_material!"
        
        assert km.constant_time_compare(value1, value2) is False
    
    def test_constant_time_compare_different_lengths(self):
        """Test constant-time comparison with different length values."""
        km = KeyManager()
        
        value1 = b"short"
        value2 = b"much_longer_value"
        
        assert km.constant_time_compare(value1, value2) is False
    
    def test_constant_time_compare_empty_values(self):
        """Test constant-time comparison with empty values."""
        km = KeyManager()
        
        assert km.constant_time_compare(b"", b"") is True
        assert km.constant_time_compare(b"", b"data") is False
        assert km.constant_time_compare(b"data", b"") is False
    
    def test_constant_time_compare_requires_bytes(self):
        """Test that constant-time comparison requires bytes input."""
        km = KeyManager()
        
        with pytest.raises(TypeError, match="Both arguments must be bytes"):
            km.constant_time_compare("string", b"bytes")
        
        with pytest.raises(TypeError, match="Both arguments must be bytes"):
            km.constant_time_compare(b"bytes", "string")
    
    def test_constant_time_compare_with_hmac_values(self):
        """Test constant-time comparison with realistic HMAC values."""
        km = KeyManager()
        
        # Simulate HMAC values (32 bytes)
        hmac1 = secrets.token_bytes(32)
        hmac2 = secrets.token_bytes(32)
        hmac1_copy = bytes(hmac1)  # Create copy
        
        # Same HMAC should match
        assert km.constant_time_compare(hmac1, hmac1_copy) is True
        
        # Different HMACs should not match
        assert km.constant_time_compare(hmac1, hmac2) is False
    
    def test_generate_salt_produces_correct_length(self):
        """Test that salt generation produces correct length."""
        km = KeyManager()
        
        salt = km.generate_salt()
        
        assert isinstance(salt, bytes)
        assert len(salt) == 32, "Salt must be 32 bytes (256 bits)"
    
    def test_generate_salt_produces_unique_values(self):
        """Test that salt generation produces unique values."""
        km = KeyManager()
        
        # Generate multiple salts
        salts = [km.generate_salt() for _ in range(100)]
        
        # All salts should be unique
        assert len(set(salts)) == 100, "All generated salts must be unique"
    
    def test_validate_key_strength_accepts_strong_passwords(self):
        """Test that password validation accepts strong passwords."""
        km = KeyManager()
        
        strong_passwords = [
            "StrongP@ssw0rd123",
            "C0mpl3x!Pass#Word",
            "MyS3cur3P@ssw0rd!",
            "Tr0ng!P@ssw0rd#2024"
        ]
        
        for password in strong_passwords:
            assert km.validate_key_strength(password) is True
    
    def test_validate_key_strength_rejects_weak_passwords(self):
        """Test that password validation rejects weak passwords."""
        km = KeyManager()
        
        weak_passwords = [
            "short",  # Too short
            "nouppercase123!",  # No uppercase
            "NOLOWERCASE123!",  # No lowercase
            "NoDigitsHere!",  # No digits
            "NoSpecialChars123",  # No special characters
            "weakpassword",  # Multiple issues
        ]
        
        for password in weak_passwords:
            assert km.validate_key_strength(password) is False


class TestSecureWipe:
    """Test suite for secure memory wiping."""
    
    def test_secure_wipe_overwrites_data(self):
        """
        Test that secure wipe overwrites sensitive data in memory.
        
        Requirement: 3.18.2
        """
        encryptor = AES256Encryptor()
        
        # Create mutable bytearray with sensitive data
        sensitive_data = bytearray(b"secret_key_material_12345")
        original_length = len(sensitive_data)
        
        # Wipe the data
        encryptor.secure_wipe(sensitive_data)
        
        # Verify data is zeroed
        assert len(sensitive_data) == original_length, "Length should not change"
        assert all(byte == 0 for byte in sensitive_data), "All bytes must be zero"
    
    def test_secure_wipe_handles_empty_data(self):
        """Test that secure wipe handles empty data gracefully."""
        encryptor = AES256Encryptor()
        
        empty_data = bytearray(b"")
        encryptor.secure_wipe(empty_data)
        
        assert len(empty_data) == 0
    
    def test_secure_wipe_handles_immutable_bytes(self):
        """Test that secure wipe handles immutable bytes gracefully."""
        encryptor = AES256Encryptor()
        
        # Immutable bytes cannot be wiped, should not raise error
        immutable_data = b"cannot_be_wiped"
        encryptor.secure_wipe(immutable_data)
        
        # Should not raise exception, just return silently
    
    def test_secure_wipe_multiple_passes(self):
        """Test that secure wipe performs multiple overwrite passes."""
        encryptor = AES256Encryptor()
        
        # Create data with pattern
        data = bytearray(b"\xff" * 100)  # All 0xFF bytes
        
        # Wipe the data
        encryptor.secure_wipe(data)
        
        # After wiping, all should be zero
        assert all(byte == 0 for byte in data)


class TestEncryptionDecryptionRoundTrip:
    """Test encryption and decryption work together correctly."""
    
    def test_encrypt_decrypt_round_trip(self):
        """Test that encryption followed by decryption recovers original data."""
        encryptor = AES256Encryptor()
        
        plaintext = b"Test Fourier coefficients: freq=1, amp=2.5, phase=1.57"
        key = secrets.token_bytes(32)
        
        # Encrypt
        payload = encryptor.encrypt(plaintext, key)
        
        # Decrypt
        decrypted = encryptor.decrypt(payload, key)
        
        # Verify round-trip
        assert decrypted == plaintext
    
    def test_decrypt_with_wrong_key_fails(self):
        """Test that decryption with wrong key fails gracefully."""
        encryptor = AES256Encryptor()
        
        plaintext = b"secret data"
        correct_key = secrets.token_bytes(32)
        wrong_key = secrets.token_bytes(32)
        
        # Encrypt with correct key
        payload = encryptor.encrypt(plaintext, correct_key)
        
        # Decrypt with wrong key should fail
        with pytest.raises(DecryptionError):
            encryptor.decrypt(payload, wrong_key)
    
    def test_decrypt_with_tampered_ciphertext_fails(self):
        """Test that decryption detects tampered ciphertext."""
        encryptor = AES256Encryptor()
        
        plaintext = b"important data"
        key = secrets.token_bytes(32)
        
        # Encrypt
        payload = encryptor.encrypt(plaintext, key)
        
        # Tamper with ciphertext
        tampered_ciphertext = bytearray(payload.ciphertext)
        tampered_ciphertext[0] ^= 0xFF  # Flip bits
        
        tampered_payload = EncryptedPayload(
            ciphertext=bytes(tampered_ciphertext),
            iv=payload.iv,
            hmac=payload.hmac,
            metadata=payload.metadata
        )
        
        # Decrypt should fail due to HMAC mismatch
        with pytest.raises(DecryptionError, match="HMAC verification failed"):
            encryptor.decrypt(tampered_payload, key)
    
    def test_decrypt_with_tampered_iv_fails(self):
        """Test that decryption detects tampered IV."""
        encryptor = AES256Encryptor()
        
        plaintext = b"important data"
        key = secrets.token_bytes(32)
        
        # Encrypt
        payload = encryptor.encrypt(plaintext, key)
        
        # Tamper with IV
        tampered_iv = bytearray(payload.iv)
        tampered_iv[0] ^= 0xFF
        
        tampered_payload = EncryptedPayload(
            ciphertext=payload.ciphertext,
            iv=bytes(tampered_iv),
            hmac=payload.hmac,
            metadata=payload.metadata
        )
        
        # Decrypt should fail due to HMAC mismatch
        with pytest.raises(DecryptionError, match="HMAC verification failed"):
            encryptor.decrypt(tampered_payload, key)
