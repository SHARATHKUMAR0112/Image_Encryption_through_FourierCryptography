"""
Unit tests for serialization module.

Tests MessagePack serialization of Fourier coefficients, metadata inclusion,
schema validation, and floating-point precision handling.

Requirements:
    - 3.5.2: Serialized data includes metadata (version, count, dimensions)
    - 3.5.3: System validates data integrity during deserialization
    - 3.5.4: Serialization handles floating-point precision consistently
"""

import pytest
import msgpack
import numpy as np

from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.models.exceptions import SerializationError


class TestCoefficientSerializer:
    """Test suite for CoefficientSerializer class."""
    
    def test_serialization_of_known_coefficient_set(self):
        """
        Test serialization of a known set of coefficients.
        
        Validates:
        - Serialization produces binary data
        - Data can be deserialized back
        - All coefficient values are preserved
        
        Requirement: 3.5.2
        """
        serializer = CoefficientSerializer()
        
        # Create known coefficient set
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=10.0,
                phase=0.0,
                complex_value=10.0+0j
            ),
            FourierCoefficient(
                frequency=1,
                amplitude=5.0,
                phase=np.pi/2,
                complex_value=5.0j
            ),
            FourierCoefficient(
                frequency=2,
                amplitude=2.5,
                phase=-np.pi/4,
                complex_value=2.5 * complex(np.cos(-np.pi/4), np.sin(-np.pi/4))
            ),
        ]
        
        metadata = {"dimensions": [512, 512]}
        
        # Serialize
        binary_data = serializer.serialize(coefficients, metadata)
        
        # Verify binary data
        assert isinstance(binary_data, bytes)
        assert len(binary_data) > 0
        
        # Deserialize and verify
        deserialized_coeffs, deserialized_metadata = serializer.deserialize(binary_data)
        
        assert len(deserialized_coeffs) == len(coefficients)
        for original, deserialized in zip(coefficients, deserialized_coeffs):
            assert deserialized.frequency == original.frequency
            assert abs(deserialized.amplitude - original.amplitude) < 1e-10
            assert abs(deserialized.phase - original.phase) < 1e-10
    
    def test_metadata_inclusion(self):
        """
        Test that serialized data includes all required metadata.
        
        Validates:
        - Version field is present
        - Coefficient count is included
        - Image dimensions are preserved
        
        Requirement: 3.5.2
        """
        serializer = CoefficientSerializer()
        
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=1.0,
                phase=0.0,
                complex_value=1.0+0j
            )
        ]
        
        metadata = {"dimensions": [1920, 1080]}
        
        # Serialize
        binary_data = serializer.serialize(coefficients, metadata)
        
        # Unpack MessagePack to inspect structure
        parsed = msgpack.unpackb(binary_data, raw=False)
        
        # Verify metadata fields
        assert "version" in parsed
        assert parsed["version"] == "1.0"
        
        assert "count" in parsed
        assert parsed["count"] == 1
        
        assert "dimensions" in parsed
        assert parsed["dimensions"] == [1920, 1080]
        
        assert "coefficients" in parsed
        assert len(parsed["coefficients"]) == 1
    
    def test_schema_validation_with_missing_fields(self):
        """
        Test schema validation rejects data with missing required fields.
        
        Validates:
        - Missing 'version' field is detected
        - Missing 'count' field is detected
        - Missing 'dimensions' field is detected
        - Missing 'coefficients' field is detected
        
        Requirement: 3.5.3
        """
        serializer = CoefficientSerializer()
        
        # Valid data structure
        valid_data = {
            "version": "1.0",
            "count": 1,
            "dimensions": [100, 100],
            "coefficients": [{"freq": 0, "amp": 1.0, "phase": 0.0}]
        }
        
        # Test missing each required field
        for field in ["version", "count", "dimensions", "coefficients"]:
            invalid_data = valid_data.copy()
            del invalid_data[field]
            
            assert not serializer.validate_schema(invalid_data), \
                f"Schema validation should fail when '{field}' is missing"
    
    def test_schema_validation_with_invalid_coefficient_fields(self):
        """
        Test schema validation rejects coefficients with missing fields.
        
        Validates:
        - Missing 'freq' field is detected
        - Missing 'amp' field is detected
        - Missing 'phase' field is detected
        
        Requirement: 3.5.3
        """
        serializer = CoefficientSerializer()
        
        # Test missing each coefficient field
        for missing_field in ["freq", "amp", "phase"]:
            coeff = {"freq": 0, "amp": 1.0, "phase": 0.0}
            del coeff[missing_field]
            
            invalid_data = {
                "version": "1.0",
                "count": 1,
                "dimensions": [100, 100],
                "coefficients": [coeff]
            }
            
            assert not serializer.validate_schema(invalid_data), \
                f"Schema validation should fail when coefficient '{missing_field}' is missing"
    
    def test_floating_point_precision_handling(self):
        """
        Test that floating-point values are handled consistently.
        
        Validates:
        - High-precision floating-point values are preserved
        - Amplitude and phase maintain precision through serialization
        - No significant precision loss occurs
        
        Requirement: 3.5.4
        """
        serializer = CoefficientSerializer()
        
        # Create coefficients with high-precision values
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=3.141592653589793,  # High precision
                phase=1.5707963267948966,     # π/2 with high precision
                complex_value=3.141592653589793 * complex(
                    np.cos(1.5707963267948966),
                    np.sin(1.5707963267948966)
                )
            ),
            FourierCoefficient(
                frequency=1,
                amplitude=2.718281828459045,  # e with high precision
                phase=-0.7853981633974483,    # -π/4 with high precision
                complex_value=2.718281828459045 * complex(
                    np.cos(-0.7853981633974483),
                    np.sin(-0.7853981633974483)
                )
            ),
        ]
        
        metadata = {"dimensions": [256, 256]}
        
        # Serialize and deserialize
        binary_data = serializer.serialize(coefficients, metadata)
        deserialized_coeffs, _ = serializer.deserialize(binary_data)
        
        # Verify precision is maintained (within floating-point tolerance)
        for original, deserialized in zip(coefficients, deserialized_coeffs):
            # Check amplitude precision
            amplitude_error = abs(deserialized.amplitude - original.amplitude)
            assert amplitude_error < 1e-10, \
                f"Amplitude precision loss: {amplitude_error}"
            
            # Check phase precision
            phase_error = abs(deserialized.phase - original.phase)
            assert phase_error < 1e-10, \
                f"Phase precision loss: {phase_error}"
    
    def test_empty_coefficients_list_raises_error(self):
        """
        Test that serializing empty coefficients list raises error.
        
        Validates:
        - Empty list is rejected
        - Appropriate error message is provided
        
        Requirement: 3.5.3
        """
        serializer = CoefficientSerializer()
        
        with pytest.raises(SerializationError) as exc_info:
            serializer.serialize([], {"dimensions": [100, 100]})
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_invalid_metadata_raises_error(self):
        """
        Test that invalid metadata raises appropriate errors.
        
        Validates:
        - Non-dict metadata is rejected
        - Missing 'dimensions' key is detected
        
        Requirement: 3.5.3
        """
        serializer = CoefficientSerializer()
        
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=1.0,
                phase=0.0,
                complex_value=1.0+0j
            )
        ]
        
        # Test non-dict metadata
        with pytest.raises(SerializationError) as exc_info:
            serializer.serialize(coefficients, "not a dict")
        
        assert "dictionary" in str(exc_info.value).lower()
        
        # Test missing dimensions key
        with pytest.raises(SerializationError) as exc_info:
            serializer.serialize(coefficients, {"other_key": "value"})
        
        assert "dimensions" in str(exc_info.value).lower()
    
    def test_corrupted_data_raises_error(self):
        """
        Test that corrupted binary data raises SerializationError.
        
        Validates:
        - Invalid MessagePack data is detected
        - Appropriate error is raised
        
        Requirement: 3.5.3
        """
        serializer = CoefficientSerializer()
        
        # Create corrupted data (invalid MessagePack)
        corrupted_data = b'\xff\xff\xff\xff\xff'
        
        with pytest.raises(SerializationError) as exc_info:
            serializer.deserialize(corrupted_data)
        
        assert "deserialize" in str(exc_info.value).lower()
    
    def test_coefficient_count_mismatch_raises_error(self):
        """
        Test that count mismatch between metadata and actual coefficients is detected.
        
        Validates:
        - Count field validation works
        - Mismatch is detected during deserialization
        
        Requirement: 3.5.3
        """
        serializer = CoefficientSerializer()
        
        # Create data with mismatched count
        invalid_data = {
            "version": "1.0",
            "count": 5,  # Claims 5 coefficients
            "dimensions": [100, 100],
            "coefficients": [  # But only has 2
                {"freq": 0, "amp": 1.0, "phase": 0.0},
                {"freq": 1, "amp": 0.5, "phase": 1.57}
            ]
        }
        
        binary_data = msgpack.packb(invalid_data, use_bin_type=True)
        
        with pytest.raises(SerializationError) as exc_info:
            serializer.deserialize(binary_data)
        
        assert "count" in str(exc_info.value).lower() or "mismatch" in str(exc_info.value).lower()
    
    def test_large_coefficient_set(self):
        """
        Test serialization of a large coefficient set.
        
        Validates:
        - System handles many coefficients efficiently
        - All coefficients are preserved
        
        Requirement: 3.5.4
        """
        serializer = CoefficientSerializer()
        
        # Create 500 coefficients
        coefficients = []
        for i in range(500):
            amplitude = 100.0 / (i + 1)  # Decreasing amplitude
            phase = (i * 0.1) % (2 * np.pi) - np.pi  # Phase in [-π, π]
            
            coefficients.append(
                FourierCoefficient(
                    frequency=i,
                    amplitude=amplitude,
                    phase=phase,
                    complex_value=amplitude * complex(np.cos(phase), np.sin(phase))
                )
            )
        
        metadata = {"dimensions": [1920, 1080]}
        
        # Serialize and deserialize
        binary_data = serializer.serialize(coefficients, metadata)
        deserialized_coeffs, deserialized_metadata = serializer.deserialize(binary_data)
        
        # Verify all coefficients preserved
        assert len(deserialized_coeffs) == 500
        assert deserialized_metadata["dimensions"] == [1920, 1080]
        
        # Spot check a few coefficients
        for i in [0, 100, 250, 499]:
            assert deserialized_coeffs[i].frequency == coefficients[i].frequency
            assert abs(deserialized_coeffs[i].amplitude - coefficients[i].amplitude) < 1e-9
            assert abs(deserialized_coeffs[i].phase - coefficients[i].phase) < 1e-9
    
    def test_deserialize_empty_data_raises_error(self):
        """
        Test that deserializing empty data raises error.
        
        Validates:
        - Empty bytes are rejected
        - Appropriate error message is provided
        
        Requirement: 3.5.3
        """
        serializer = CoefficientSerializer()
        
        with pytest.raises(SerializationError) as exc_info:
            serializer.deserialize(b'')
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_deserialize_non_bytes_raises_error(self):
        """
        Test that deserializing non-bytes data raises error.
        
        Validates:
        - Type validation works
        - Appropriate error message is provided
        
        Requirement: 3.5.3
        """
        serializer = CoefficientSerializer()
        
        with pytest.raises(SerializationError) as exc_info:
            serializer.deserialize("not bytes")
        
        assert "bytes" in str(exc_info.value).lower()
