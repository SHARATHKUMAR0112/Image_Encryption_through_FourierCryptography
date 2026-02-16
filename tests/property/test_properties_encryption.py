"""
Property-based tests for Encryption Layer.

Feature: fourier-image-encryption
Property 9: IV Uniqueness
Property 10: HMAC Integrity Validation
Property 11: Encryption Round-Trip
Property 12: Wrong Key Rejection

These tests verify the cryptographic correctness and security properties
of the AES-256-GCM encryption implementation.
"""

import secrets
import pytest
from hypothesis import given, strategies as st, assume

from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.models.data_models import EncryptedPayload
from fourier_encryption.models.exceptions import DecryptionError


# Property 9: IV Uniqueness
@given(
    data=st.binary(min_size=1, max_size=1000),
)
@pytest.mark.property_test
def test_iv_uniqueness_same_data_same_key(data):
    """
    Feature: fourier-image-encryption
    Property 9: IV Uniqueness
    
    For any two independent encryption operations (even with the same input
    and key), the generated initialization vectors (IVs) must be different
    with overwhelming probability (collision probability < 2^-128).
    
    **Validates: Requirements 3.4.3**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)  # 256-bit key
    
    # Encrypt the same data twice with the same key
    payload1 = encryptor.encrypt(data, key)
    payload2 = encryptor.encrypt(data, key)
    
    # IVs must be different
    assert payload1.iv != payload2.iv, (
        "Two independent encryptions produced the same IV, "
        "which violates cryptographic security requirements"
    )
    
    # Verify IV lengths are correct
    assert len(payload1.iv) == 16
    assert len(payload2.iv) == 16


@given(
    data1=st.binary(min_size=1, max_size=500),
    data2=st.binary(min_size=1, max_size=500),
)
@pytest.mark.property_test
def test_iv_uniqueness_different_data_same_key(data1, data2):
    """
    Feature: fourier-image-encryption
    Property 9: IV Uniqueness (different data)
    
    For any two encryption operations with different data but the same key,
    the IVs must be different.
    
    **Validates: Requirements 3.4.3**
    """
    # Ensure data is different
    assume(data1 != data2)
    
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    
    payload1 = encryptor.encrypt(data1, key)
    payload2 = encryptor.encrypt(data2, key)
    
    # IVs must be different
    assert payload1.iv != payload2.iv


@pytest.mark.property_test
def test_iv_uniqueness_multiple_encryptions():
    """
    Feature: fourier-image-encryption
    Property 9: IV Uniqueness (multiple encryptions)
    
    For multiple encryption operations, all IVs should be unique.
    
    **Validates: Requirements 3.4.3**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    data = b"test data"
    
    # Perform 100 encryptions
    num_encryptions = 100
    ivs = set()
    
    for _ in range(num_encryptions):
        payload = encryptor.encrypt(data, key)
        ivs.add(payload.iv)
    
    # All IVs should be unique
    assert len(ivs) == num_encryptions, (
        f"Expected {num_encryptions} unique IVs, but got {len(ivs)}"
    )


# Property 10: HMAC Integrity Validation
@given(
    data=st.binary(min_size=1, max_size=1000),
    tamper_position=st.integers(min_value=0, max_value=999),
)
@pytest.mark.property_test
def test_hmac_detects_ciphertext_tampering(data, tamper_position):
    """
    Feature: fourier-image-encryption
    Property 10: HMAC Integrity Validation
    
    For any encrypted payload, if the ciphertext is modified by even a
    single bit, HMAC verification must fail and decryption must raise
    an integrity error.
    
    **Validates: Requirements 3.4.4**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    
    # Encrypt data
    payload = encryptor.encrypt(data, key)
    
    # Tamper with ciphertext at a valid position
    if len(payload.ciphertext) > 0:
        tamper_position = tamper_position % len(payload.ciphertext)
        
        # Flip one bit in the ciphertext
        tampered_ciphertext = bytearray(payload.ciphertext)
        tampered_ciphertext[tamper_position] ^= 0x01
        
        # Create tampered payload
        tampered_payload = EncryptedPayload(
            ciphertext=bytes(tampered_ciphertext),
            iv=payload.iv,
            hmac=payload.hmac,
            metadata=payload.metadata
        )
        
        # Decryption must fail with DecryptionError
        with pytest.raises(DecryptionError) as exc_info:
            encryptor.decrypt(tampered_payload, key)
        
        # Error message should indicate HMAC verification failure
        assert "HMAC verification failed" in str(exc_info.value)


@given(
    data=st.binary(min_size=1, max_size=1000),
    tamper_position=st.integers(min_value=0, max_value=15),
)
@pytest.mark.property_test
def test_hmac_detects_iv_tampering(data, tamper_position):
    """
    Feature: fourier-image-encryption
    Property 10: HMAC Integrity Validation (IV tampering)
    
    For any encrypted payload, if the IV is modified, HMAC verification
    must fail.
    
    **Validates: Requirements 3.4.4**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    
    # Encrypt data
    payload = encryptor.encrypt(data, key)
    
    # Tamper with IV
    tampered_iv = bytearray(payload.iv)
    tampered_iv[tamper_position] ^= 0x01
    
    # Create tampered payload
    tampered_payload = EncryptedPayload(
        ciphertext=payload.ciphertext,
        iv=bytes(tampered_iv),
        hmac=payload.hmac,
        metadata=payload.metadata
    )
    
    # Decryption must fail
    with pytest.raises(DecryptionError) as exc_info:
        encryptor.decrypt(tampered_payload, key)
    
    assert "HMAC verification failed" in str(exc_info.value)


@given(
    data=st.binary(min_size=1, max_size=1000),
)
@pytest.mark.property_test
def test_hmac_detects_hmac_tampering(data):
    """
    Feature: fourier-image-encryption
    Property 10: HMAC Integrity Validation (HMAC tampering)
    
    For any encrypted payload, if the HMAC itself is modified,
    verification must fail.
    
    **Validates: Requirements 3.4.4**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    
    # Encrypt data
    payload = encryptor.encrypt(data, key)
    
    # Replace HMAC with random bytes
    tampered_hmac = secrets.token_bytes(32)
    
    # Ensure it's different from the original
    assume(tampered_hmac != payload.hmac)
    
    # Create tampered payload
    tampered_payload = EncryptedPayload(
        ciphertext=payload.ciphertext,
        iv=payload.iv,
        hmac=tampered_hmac,
        metadata=payload.metadata
    )
    
    # Decryption must fail
    with pytest.raises(DecryptionError) as exc_info:
        encryptor.decrypt(tampered_payload, key)
    
    assert "HMAC verification failed" in str(exc_info.value)


@pytest.mark.property_test
def test_hmac_validates_untampered_data():
    """
    Feature: fourier-image-encryption
    Property 10: HMAC Integrity Validation (valid data)
    
    For untampered encrypted payloads, HMAC verification should succeed
    and decryption should complete successfully.
    
    **Validates: Requirements 3.4.4**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    data = b"test data that should decrypt successfully"
    
    # Encrypt data
    payload = encryptor.encrypt(data, key)
    
    # Decryption should succeed without raising any errors
    decrypted = encryptor.decrypt(payload, key)
    
    assert decrypted == data


# Property 11: Encryption Round-Trip
@given(
    data=st.binary(min_size=1, max_size=10000),
)
@pytest.mark.property_test
def test_encryption_round_trip(data):
    """
    Feature: fourier-image-encryption
    Property 11: Encryption Round-Trip
    
    For any data and encryption key, encrypting then decrypting with the
    same key should recover the original data exactly.
    
    **Validates: Requirements 3.4.5**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    
    # Encrypt
    payload = encryptor.encrypt(data, key)
    
    # Verify payload structure
    assert isinstance(payload, EncryptedPayload)
    assert len(payload.iv) == 16
    assert len(payload.hmac) == 32
    assert len(payload.ciphertext) > 0
    
    # Decrypt
    decrypted = encryptor.decrypt(payload, key)
    
    # Verify exact recovery
    assert decrypted == data


@given(
    data=st.binary(min_size=0, max_size=5000),
)
@pytest.mark.property_test
def test_encryption_round_trip_various_sizes(data):
    """
    Feature: fourier-image-encryption
    Property 11: Encryption Round-Trip (various sizes)
    
    Encryption round-trip should work for data of any size, including
    empty data.
    
    **Validates: Requirements 3.4.5**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    
    payload = encryptor.encrypt(data, key)
    decrypted = encryptor.decrypt(payload, key)
    
    assert decrypted == data


@given(
    data=st.binary(min_size=1, max_size=1000),
    kdf_iterations=st.integers(min_value=100_000, max_value=200_000),
)
@pytest.mark.property_test
def test_encryption_round_trip_different_kdf_iterations(data, kdf_iterations):
    """
    Feature: fourier-image-encryption
    Property 11: Encryption Round-Trip (different KDF iterations)
    
    Encryption round-trip should work with different KDF iteration counts.
    
    **Validates: Requirements 3.4.5**
    """
    encryptor = AES256Encryptor(kdf_iterations=kdf_iterations)
    key = secrets.token_bytes(32)
    
    payload = encryptor.encrypt(data, key)
    decrypted = encryptor.decrypt(payload, key)
    
    assert decrypted == data
    assert payload.metadata["kdf_iterations"] == kdf_iterations


@pytest.mark.property_test
def test_encryption_round_trip_preserves_metadata():
    """
    Feature: fourier-image-encryption
    Property 11: Encryption Round-Trip (metadata preservation)
    
    Encryption should include proper metadata that is preserved through
    the round-trip.
    
    **Validates: Requirements 3.4.5**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    data = b"test data with metadata"
    
    payload = encryptor.encrypt(data, key)
    
    # Verify metadata is present
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
    assert payload.metadata["data_length"] == len(data)
    
    # Decrypt should succeed
    decrypted = encryptor.decrypt(payload, key)
    assert decrypted == data


# Property 12: Wrong Key Rejection
@given(
    data=st.binary(min_size=1, max_size=1000),
)
@pytest.mark.property_test
def test_wrong_key_rejection(data):
    """
    Feature: fourier-image-encryption
    Property 12: Wrong Key Rejection
    
    For any encrypted payload and incorrect decryption key, the decryption
    operation must fail gracefully with a DecryptionError exception
    (not crash or return corrupted data).
    
    **Validates: Requirements 3.4.6**
    """
    encryptor = AES256Encryptor()
    correct_key = secrets.token_bytes(32)
    wrong_key = secrets.token_bytes(32)
    
    # Ensure keys are different
    assume(correct_key != wrong_key)
    
    # Encrypt with correct key
    payload = encryptor.encrypt(data, correct_key)
    
    # Attempt to decrypt with wrong key must raise DecryptionError
    with pytest.raises(DecryptionError) as exc_info:
        encryptor.decrypt(payload, wrong_key)
    
    # Verify it's a DecryptionError, not a generic exception
    assert isinstance(exc_info.value, DecryptionError)
    
    # Error message should indicate the problem
    error_msg = str(exc_info.value)
    assert "HMAC verification failed" in error_msg or "Decryption failed" in error_msg


@given(
    data=st.binary(min_size=1, max_size=1000),
)
@pytest.mark.property_test
def test_wrong_key_rejection_multiple_attempts(data):
    """
    Feature: fourier-image-encryption
    Property 12: Wrong Key Rejection (multiple attempts)
    
    Multiple decryption attempts with wrong keys should all fail gracefully.
    
    **Validates: Requirements 3.4.6**
    """
    encryptor = AES256Encryptor()
    correct_key = secrets.token_bytes(32)
    
    # Encrypt with correct key
    payload = encryptor.encrypt(data, correct_key)
    
    # Try multiple wrong keys
    for _ in range(10):
        wrong_key = secrets.token_bytes(32)
        
        if wrong_key != correct_key:
            with pytest.raises(DecryptionError):
                encryptor.decrypt(payload, wrong_key)


@given(
    data=st.binary(min_size=1, max_size=1000),
)
@pytest.mark.property_test
def test_wrong_key_rejection_slightly_modified_key(data):
    """
    Feature: fourier-image-encryption
    Property 12: Wrong Key Rejection (slightly modified key)
    
    Even a single bit difference in the key should cause decryption to fail.
    
    **Validates: Requirements 3.4.6**
    """
    encryptor = AES256Encryptor()
    correct_key = secrets.token_bytes(32)
    
    # Encrypt with correct key
    payload = encryptor.encrypt(data, correct_key)
    
    # Create a key with one bit flipped
    wrong_key = bytearray(correct_key)
    wrong_key[0] ^= 0x01  # Flip one bit
    wrong_key = bytes(wrong_key)
    
    # Decryption must fail
    with pytest.raises(DecryptionError):
        encryptor.decrypt(payload, wrong_key)


@pytest.mark.property_test
def test_wrong_key_rejection_does_not_crash():
    """
    Feature: fourier-image-encryption
    Property 12: Wrong Key Rejection (graceful failure)
    
    Decryption with wrong key should fail gracefully without crashing
    or raising unexpected exceptions.
    
    **Validates: Requirements 3.4.6**
    """
    encryptor = AES256Encryptor()
    correct_key = secrets.token_bytes(32)
    wrong_key = secrets.token_bytes(32)
    data = b"sensitive data"
    
    payload = encryptor.encrypt(data, correct_key)
    
    # Should raise DecryptionError, not crash with other exceptions
    try:
        encryptor.decrypt(payload, wrong_key)
        # If we get here, decryption succeeded when it shouldn't have
        pytest.fail("Decryption with wrong key should have raised DecryptionError")
    except DecryptionError:
        # This is expected - graceful failure
        pass
    except Exception as e:
        # Any other exception is a test failure
        pytest.fail(
            f"Expected DecryptionError, but got {type(e).__name__}: {e}"
        )


@given(
    data=st.binary(min_size=1, max_size=1000),
)
@pytest.mark.property_test
def test_correct_key_always_succeeds(data):
    """
    Feature: fourier-image-encryption
    Property 12: Wrong Key Rejection (correct key succeeds)
    
    As a sanity check, decryption with the correct key should always succeed.
    
    **Validates: Requirements 3.4.6**
    """
    encryptor = AES256Encryptor()
    key = secrets.token_bytes(32)
    
    payload = encryptor.encrypt(data, key)
    
    # Decryption with correct key should succeed
    decrypted = encryptor.decrypt(payload, key)
    assert decrypted == data
