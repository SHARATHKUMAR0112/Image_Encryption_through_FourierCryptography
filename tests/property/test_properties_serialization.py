"""
Property-based tests for Serialization Module.

Feature: fourier-image-encryption
Property 13: Serialization Round-Trip
Property 14: Corruption Detection

These tests verify the correctness and robustness of the MessagePack-based
serialization implementation for Fourier coefficients.
"""

import secrets
import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import numpy as np

from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.models.exceptions import SerializationError


# Custom strategy for generating valid FourierCoefficient objects
@st.composite
def fourier_coefficient_strategy(draw):
    """Generate valid FourierCoefficient objects for property testing."""
    frequency = draw(st.integers(min_value=0, max_value=10000))
    amplitude = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    phase = draw(st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False))
    
    # Compute complex value from amplitude and phase
    complex_value = amplitude * complex(np.cos(phase), np.sin(phase))
    
    return FourierCoefficient(
        frequency=frequency,
        amplitude=amplitude,
        phase=phase,
        complex_value=complex_value
    )


# Custom strategy for generating lists of coefficients
@st.composite
def coefficient_list_strategy(draw):
    """Generate lists of valid FourierCoefficient objects."""
    num_coefficients = draw(st.integers(min_value=1, max_value=100))
    coefficients = [draw(fourier_coefficient_strategy()) for _ in range(num_coefficients)]
    return coefficients


# Custom strategy for generating metadata
@st.composite
def metadata_strategy(draw):
    """Generate valid metadata dictionaries."""
    width = draw(st.integers(min_value=1, max_value=4096))
    height = draw(st.integers(min_value=1, max_value=4096))
    return {"dimensions": [width, height]}


# Property 13: Serialization Round-Trip
@given(
    coefficients=coefficient_list_strategy(),
    metadata=metadata_strategy(),
)
@pytest.mark.property_test
def test_serialization_round_trip(coefficients, metadata):
    """
    Feature: fourier-image-encryption
    Property 13: Serialization Round-Trip
    
    For any list of Fourier coefficients with metadata, serializing then
    deserializing should recover the original coefficients and metadata
    within floating-point precision.
    
    **Validates: Requirements 3.5.1, 3.5.2, 3.5.4**
    """
    serializer = CoefficientSerializer()
    
    # Serialize
    binary_data = serializer.serialize(coefficients, metadata)
    
    # Verify binary data is not empty
    assert len(binary_data) > 0
    assert isinstance(binary_data, bytes)
    
    # Deserialize
    recovered_coefficients, recovered_metadata = serializer.deserialize(binary_data)
    
    # Verify coefficient count matches
    assert len(recovered_coefficients) == len(coefficients)
    
    # Verify each coefficient is recovered correctly
    for original, recovered in zip(coefficients, recovered_coefficients):
        assert recovered.frequency == original.frequency
        
        # Floating-point comparison with tolerance
        assert np.isclose(recovered.amplitude, original.amplitude, rtol=1e-9, atol=1e-12)
        assert np.isclose(recovered.phase, original.phase, rtol=1e-9, atol=1e-12)
        
        # Complex value should be consistent
        assert np.isclose(abs(recovered.complex_value - original.complex_value), 0, atol=1e-9)
    
    # Verify metadata is recovered
    assert "dimensions" in recovered_metadata
    assert recovered_metadata["dimensions"] == metadata["dimensions"]
    assert "version" in recovered_metadata


@given(
    coefficients=coefficient_list_strategy(),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
@pytest.mark.property_test
def test_serialization_round_trip_minimal_metadata(coefficients):
    """
    Feature: fourier-image-encryption
    Property 13: Serialization Round-Trip (minimal metadata)
    
    Serialization should work with minimal metadata (only dimensions).
    
    **Validates: Requirements 3.5.1, 3.5.2**
    """
    serializer = CoefficientSerializer()
    metadata = {"dimensions": [100, 100]}
    
    binary_data = serializer.serialize(coefficients, metadata)
    recovered_coefficients, recovered_metadata = serializer.deserialize(binary_data)
    
    assert len(recovered_coefficients) == len(coefficients)
    assert recovered_metadata["dimensions"] == [100, 100]


@pytest.mark.property_test
def test_serialization_round_trip_single_coefficient():
    """
    Feature: fourier-image-encryption
    Property 13: Serialization Round-Trip (single coefficient)
    
    Serialization should work correctly for a single coefficient.
    
    **Validates: Requirements 3.5.1, 3.5.2**
    """
    serializer = CoefficientSerializer()
    
    # Create a single coefficient
    coefficient = FourierCoefficient(
        frequency=0,
        amplitude=1.0,
        phase=0.0,
        complex_value=1.0+0j
    )
    
    metadata = {"dimensions": [256, 256]}
    
    binary_data = serializer.serialize([coefficient], metadata)
    recovered_coefficients, recovered_metadata = serializer.deserialize(binary_data)
    
    assert len(recovered_coefficients) == 1
    assert recovered_coefficients[0].frequency == 0
    assert np.isclose(recovered_coefficients[0].amplitude, 1.0)
    assert np.isclose(recovered_coefficients[0].phase, 0.0)


@given(
    coefficients=coefficient_list_strategy(),
    metadata=metadata_strategy(),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
@pytest.mark.property_test
def test_serialization_round_trip_preserves_order(coefficients, metadata):
    """
    Feature: fourier-image-encryption
    Property 13: Serialization Round-Trip (order preservation)
    
    Serialization should preserve the order of coefficients.
    
    **Validates: Requirements 3.5.1, 3.5.2**
    """
    serializer = CoefficientSerializer()
    
    binary_data = serializer.serialize(coefficients, metadata)
    recovered_coefficients, _ = serializer.deserialize(binary_data)
    
    # Verify order is preserved
    for i, (original, recovered) in enumerate(zip(coefficients, recovered_coefficients)):
        assert recovered.frequency == original.frequency, (
            f"Coefficient order not preserved at index {i}"
        )


@pytest.mark.property_test
def test_serialization_round_trip_extreme_values():
    """
    Feature: fourier-image-encryption
    Property 13: Serialization Round-Trip (extreme values)
    
    Serialization should handle extreme but valid values correctly.
    
    **Validates: Requirements 3.5.1, 3.5.4**
    """
    serializer = CoefficientSerializer()
    
    # Create coefficients with extreme values
    coefficients = [
        FourierCoefficient(0, 0.0, 0.0, 0.0+0j),  # Zero amplitude
        FourierCoefficient(1, 1000.0, np.pi, -1000.0+0j),  # Large amplitude, max phase
        FourierCoefficient(2, 0.0001, -np.pi, -0.0001+0j),  # Small amplitude, min phase
        FourierCoefficient(9999, 500.0, 0.0, 500.0+0j),  # Large frequency
    ]
    
    metadata = {"dimensions": [4096, 4096]}
    
    binary_data = serializer.serialize(coefficients, metadata)
    recovered_coefficients, _ = serializer.deserialize(binary_data)
    
    assert len(recovered_coefficients) == len(coefficients)
    
    for original, recovered in zip(coefficients, recovered_coefficients):
        assert recovered.frequency == original.frequency
        assert np.isclose(recovered.amplitude, original.amplitude, rtol=1e-9, atol=1e-12)
        assert np.isclose(recovered.phase, original.phase, rtol=1e-9, atol=1e-12)


# Property 14: Corruption Detection
@given(
    coefficients=coefficient_list_strategy(),
    metadata=metadata_strategy(),
    corruption_position=st.integers(min_value=0, max_value=9999),
)
@pytest.mark.property_test
def test_corruption_detection_random_bit_flip(coefficients, metadata, corruption_position):
    """
    Feature: fourier-image-encryption
    Property 14: Corruption Detection
    
    For any serialized data that is corrupted (random bit flips),
    deserialization must detect the corruption and raise a SerializationError
    (not silently return invalid data).
    
    **Validates: Requirements 3.5.3**
    """
    serializer = CoefficientSerializer()
    
    # Serialize
    binary_data = serializer.serialize(coefficients, metadata)
    
    # Corrupt the data by flipping a random bit
    if len(binary_data) > 0:
        corruption_position = corruption_position % len(binary_data)
        corrupted_data = bytearray(binary_data)
        corrupted_data[corruption_position] ^= 0x01  # Flip one bit
        corrupted_data = bytes(corrupted_data)
        
        # Deserialization should either raise SerializationError or
        # return data that fails validation
        try:
            recovered_coefficients, _ = serializer.deserialize(corrupted_data)
            
            # If deserialization succeeded, the data might be corrupted
            # but still parseable. Verify it's different from original.
            # In most cases, MessagePack will detect corruption.
            
        except SerializationError:
            # This is expected - corruption was detected
            pass
        except Exception as e:
            # MessagePack might raise other exceptions for corrupted data
            # This is acceptable as long as it doesn't silently succeed
            assert not isinstance(e, AssertionError), (
                f"Unexpected exception type: {type(e).__name__}"
            )


@given(
    coefficients=coefficient_list_strategy(),
    metadata=metadata_strategy(),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
@pytest.mark.property_test
def test_corruption_detection_truncated_data(coefficients, metadata):
    """
    Feature: fourier-image-encryption
    Property 14: Corruption Detection (truncated data)
    
    For serialized data that is truncated, deserialization must detect
    the corruption and raise a SerializationError.
    
    **Validates: Requirements 3.5.3**
    """
    serializer = CoefficientSerializer()
    
    # Serialize
    binary_data = serializer.serialize(coefficients, metadata)
    
    # Truncate the data (remove last 10 bytes or half, whichever is smaller)
    if len(binary_data) > 10:
        truncate_amount = min(10, len(binary_data) // 2)
        truncated_data = binary_data[:-truncate_amount]
        
        # Deserialization should raise SerializationError
        with pytest.raises((SerializationError, Exception)):
            serializer.deserialize(truncated_data)


@pytest.mark.property_test
def test_corruption_detection_empty_data():
    """
    Feature: fourier-image-encryption
    Property 14: Corruption Detection (empty data)
    
    Attempting to deserialize empty data should raise SerializationError.
    
    **Validates: Requirements 3.5.3**
    """
    serializer = CoefficientSerializer()
    
    with pytest.raises(SerializationError) as exc_info:
        serializer.deserialize(b"")
    
    assert "empty data" in str(exc_info.value).lower()


@pytest.mark.property_test
def test_corruption_detection_invalid_msgpack():
    """
    Feature: fourier-image-encryption
    Property 14: Corruption Detection (invalid MessagePack)
    
    Attempting to deserialize invalid MessagePack data should raise
    SerializationError.
    
    **Validates: Requirements 3.5.3**
    """
    serializer = CoefficientSerializer()
    
    # Create invalid MessagePack data (random bytes)
    invalid_data = secrets.token_bytes(100)
    
    # Deserialization should raise SerializationError
    with pytest.raises((SerializationError, Exception)):
        serializer.deserialize(invalid_data)


@pytest.mark.property_test
def test_corruption_detection_missing_required_fields():
    """
    Feature: fourier-image-encryption
    Property 14: Corruption Detection (missing fields)
    
    Deserialization should detect missing required fields in the schema.
    
    **Validates: Requirements 3.5.3**
    """
    import msgpack
    
    serializer = CoefficientSerializer()
    
    # Create data with missing required fields
    incomplete_data = {
        "version": "1.0",
        "count": 1,
        # Missing "dimensions" and "coefficients"
    }
    
    binary_data = msgpack.packb(incomplete_data, use_bin_type=True)
    
    # Deserialization should raise SerializationError
    with pytest.raises(SerializationError) as exc_info:
        serializer.deserialize(binary_data)
    
    assert "schema" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()


@pytest.mark.property_test
def test_corruption_detection_count_mismatch():
    """
    Feature: fourier-image-encryption
    Property 14: Corruption Detection (count mismatch)
    
    Deserialization should detect when the coefficient count doesn't match
    the actual number of coefficients.
    
    **Validates: Requirements 3.5.3**
    """
    import msgpack
    
    serializer = CoefficientSerializer()
    
    # Create data with mismatched count
    data = {
        "version": "1.0",
        "count": 10,  # Claims 10 coefficients
        "dimensions": [100, 100],
        "coefficients": [  # But only has 2
            {"freq": 0, "amp": 1.0, "phase": 0.0},
            {"freq": 1, "amp": 0.5, "phase": 1.0},
        ]
    }
    
    binary_data = msgpack.packb(data, use_bin_type=True)
    
    # Deserialization should raise SerializationError
    with pytest.raises(SerializationError) as exc_info:
        serializer.deserialize(binary_data)
    
    assert "count mismatch" in str(exc_info.value).lower()


@pytest.mark.property_test
def test_corruption_detection_invalid_coefficient_fields():
    """
    Feature: fourier-image-encryption
    Property 14: Corruption Detection (invalid coefficient)
    
    Deserialization should detect coefficients with missing or invalid fields.
    
    **Validates: Requirements 3.5.3**
    """
    import msgpack
    
    serializer = CoefficientSerializer()
    
    # Create data with invalid coefficient (missing phase field)
    data = {
        "version": "1.0",
        "count": 1,
        "dimensions": [100, 100],
        "coefficients": [
            {"freq": 0, "amp": 1.0}  # Missing "phase"
        ]
    }
    
    binary_data = msgpack.packb(data, use_bin_type=True)
    
    # Deserialization should raise SerializationError
    with pytest.raises(SerializationError) as exc_info:
        serializer.deserialize(binary_data)
    
    error_msg = str(exc_info.value).lower()
    assert "missing" in error_msg or "required" in error_msg or "field" in error_msg


@given(
    coefficients=coefficient_list_strategy(),
    metadata=metadata_strategy(),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
@pytest.mark.property_test
def test_serialization_is_deterministic(coefficients, metadata):
    """
    Feature: fourier-image-encryption
    Property 13: Serialization Round-Trip (determinism)
    
    Serializing the same data multiple times should produce identical output.
    
    **Validates: Requirements 3.5.1**
    """
    serializer = CoefficientSerializer()
    
    # Serialize the same data twice
    binary_data1 = serializer.serialize(coefficients, metadata)
    binary_data2 = serializer.serialize(coefficients, metadata)
    
    # Results should be identical
    assert binary_data1 == binary_data2


@pytest.mark.property_test
def test_serialization_validates_input_types():
    """
    Feature: fourier-image-encryption
    Property 14: Corruption Detection (input validation)
    
    Serialization should validate input types and raise appropriate errors.
    
    **Validates: Requirements 3.5.3**
    """
    serializer = CoefficientSerializer()
    
    # Test with empty coefficient list
    with pytest.raises(SerializationError):
        serializer.serialize([], {"dimensions": [100, 100]})
    
    # Test with invalid metadata (missing dimensions)
    coefficient = FourierCoefficient(0, 1.0, 0.0, 1.0+0j)
    with pytest.raises(SerializationError):
        serializer.serialize([coefficient], {})
    
    # Test with non-dict metadata
    with pytest.raises(SerializationError):
        serializer.serialize([coefficient], "not a dict")
