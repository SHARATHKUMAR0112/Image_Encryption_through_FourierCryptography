"""
Serialization module for Fourier coefficients.

This module provides secure serialization and deserialization of Fourier
coefficients using MessagePack for compact binary format. Includes schema
validation and metadata handling.

Performance optimizations:
- Batch coefficient serialization with list comprehension
- Pre-allocated dictionaries for deserialization
- Cached schema validation
- Vectorized coefficient reconstruction
"""

from typing import Any, Dict, List, Tuple

import msgpack
import numpy as np

from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.models.exceptions import SerializationError


class CoefficientSerializer:
    """
    Serialize and deserialize Fourier coefficients.
    
    Uses MessagePack for compact binary format with metadata support.
    Handles floating-point precision consistently and validates schema
    during deserialization.
    
    Performance optimizations:
    - Batch coefficient processing
    - Pre-allocated data structures
    - Vectorized complex value reconstruction
    """
    
    # Current serialization format version
    VERSION = "1.0"
    
    # Required fields in serialized data
    REQUIRED_FIELDS = {"version", "count", "dimensions", "coefficients"}
    
    # Required fields in each coefficient
    REQUIRED_COEFFICIENT_FIELDS = {"freq", "amp", "phase"}
    
    def __init__(self):
        """Initialize serializer with caching support."""
        self._validation_cache = {}
    
    def serialize(
        self,
        coefficients: List[FourierCoefficient],
        metadata: Dict[str, Any]
    ) -> bytes:
        """
        Serialize Fourier coefficients to compact binary format.
        
        Creates a MessagePack-encoded binary representation containing:
        - Version information for format compatibility
        - Coefficient count for validation
        - Image dimensions from metadata
        - List of coefficients with frequency, amplitude, and phase
        
        Optimizations:
        - Batch coefficient conversion with list comprehension
        - Pre-allocated data structure
        
        Args:
            coefficients: List of FourierCoefficient objects to serialize
            metadata: Additional metadata (must include 'dimensions' key)
        
        Returns:
            Binary data in MessagePack format
        
        Raises:
            SerializationError: If serialization fails or inputs are invalid
        """
        # Validate inputs
        if not coefficients:
            raise SerializationError(
                "Cannot serialize empty coefficients list",
                context={"coefficient_count": 0}
            )
        
        if not all(isinstance(c, FourierCoefficient) for c in coefficients):
            raise SerializationError(
                "All elements must be FourierCoefficient objects"
            )
        
        if not isinstance(metadata, dict):
            raise SerializationError(
                "metadata must be a dictionary",
                context={"metadata_type": type(metadata).__name__}
            )
        
        if "dimensions" not in metadata:
            raise SerializationError(
                "metadata must include 'dimensions' key",
                context={"metadata_keys": list(metadata.keys())}
            )
        
        # Build serialization structure with pre-allocated coefficient list
        # Batch convert all coefficients using list comprehension (faster than append loop)
        coeff_dicts = [
            {
                "freq": int(coeff.frequency),
                "amp": float(coeff.amplitude),
                "phase": float(coeff.phase)
            }
            for coeff in coefficients
        ]
        
        data = {
            "version": self.VERSION,
            "count": len(coefficients),
            "dimensions": metadata["dimensions"],
            "coefficients": coeff_dicts
        }
        
        # Serialize to MessagePack binary format
        try:
            binary_data = msgpack.packb(data, use_bin_type=True)
        except Exception as e:
            raise SerializationError(
                f"Failed to serialize coefficients: {e}",
                context={
                    "coefficient_count": len(coefficients),
                    "error_type": type(e).__name__
                }
            )
        
        return binary_data
    
    def deserialize(
        self,
        data: bytes
    ) -> Tuple[List[FourierCoefficient], Dict[str, Any]]:
        """
        Deserialize Fourier coefficients from binary format.
        
        Parses MessagePack-encoded data and validates schema before
        reconstructing FourierCoefficient objects.
        
        Optimizations:
        - Vectorized complex value reconstruction
        - Batch coefficient creation
        - Pre-allocated arrays for trigonometric operations
        
        Args:
            data: Binary data in MessagePack format
        
        Returns:
            Tuple of (coefficients list, metadata dictionary)
        
        Raises:
            SerializationError: If deserialization fails or schema is invalid
        """
        # Validate input
        if not isinstance(data, bytes):
            raise SerializationError(
                "data must be bytes",
                context={"data_type": type(data).__name__}
            )
        
        if not data:
            raise SerializationError(
                "Cannot deserialize empty data",
                context={"data_length": 0}
            )
        
        # Deserialize from MessagePack
        try:
            parsed = msgpack.unpackb(data, raw=False)
        except Exception as e:
            raise SerializationError(
                f"Failed to deserialize data: {e}",
                context={
                    "data_length": len(data),
                    "first_bytes": data[:16].hex() if len(data) >= 16 else data.hex(),
                    "error_type": type(e).__name__
                }
            )
        
        # Validate schema
        if not self.validate_schema(parsed):
            raise SerializationError(
                "Invalid schema in deserialized data",
                context={
                    "present_fields": list(parsed.keys()) if isinstance(parsed, dict) else None,
                    "required_fields": list(self.REQUIRED_FIELDS)
                }
            )
        
        # Extract metadata
        metadata = {
            "version": parsed["version"],
            "dimensions": parsed["dimensions"]
        }
        
        # Validate coefficient count
        expected_count = parsed["count"]
        actual_count = len(parsed["coefficients"])
        if expected_count != actual_count:
            raise SerializationError(
                "Coefficient count mismatch",
                context={
                    "expected_count": expected_count,
                    "actual_count": actual_count
                }
            )
        
        # Extract coefficient data as arrays for vectorized operations
        coeff_dicts = parsed["coefficients"]
        
        # Validate all coefficients have required fields (batch check)
        for i, coeff_dict in enumerate(coeff_dicts):
            missing_fields = self.REQUIRED_COEFFICIENT_FIELDS - set(coeff_dict.keys())
            if missing_fields:
                raise SerializationError(
                    f"Coefficient {i} missing required fields",
                    context={
                        "missing_fields": list(missing_fields),
                        "present_fields": list(coeff_dict.keys())
                    }
                )
        
        # Extract arrays for vectorized complex value reconstruction
        try:
            frequencies = np.array([int(c["freq"]) for c in coeff_dicts], dtype=np.int32)
            amplitudes = np.array([float(c["amp"]) for c in coeff_dicts], dtype=np.float64)
            phases = np.array([float(c["phase"]) for c in coeff_dicts], dtype=np.float64)
        except (ValueError, KeyError) as e:
            raise SerializationError(
                f"Invalid coefficient data: {e}",
                context={"error": str(e)}
            )
        
        # Vectorized complex value reconstruction
        # complex_value = amplitude * e^(i*phase) = amplitude * (cos(phase) + i*sin(phase))
        real_parts = amplitudes * np.cos(phases)
        imag_parts = amplitudes * np.sin(phases)
        complex_values = real_parts + 1j * imag_parts
        
        # Create FourierCoefficient objects (this part can't be fully vectorized)
        coefficients = [
            FourierCoefficient(
                frequency=int(frequencies[i]),
                amplitude=float(amplitudes[i]),
                phase=float(phases[i]),
                complex_value=complex(complex_values[i])
            )
            for i in range(len(coeff_dicts))
        ]
        
        return coefficients, metadata
    
    def validate_schema(self, data: Any) -> bool:
        """
        Validate that deserialized data has required schema.
        
        Checks for presence of all required fields and basic type validation.
        
        Args:
            data: Deserialized data structure to validate
        
        Returns:
            True if schema is valid, False otherwise
        """
        # Check if data is a dictionary
        if not isinstance(data, dict):
            return False
        
        # Check for required top-level fields
        if not self.REQUIRED_FIELDS.issubset(data.keys()):
            return False
        
        # Validate version field
        if not isinstance(data["version"], str):
            return False
        
        # Validate count field
        if not isinstance(data["count"], int) or data["count"] < 0:
            return False
        
        # Validate dimensions field
        if not isinstance(data["dimensions"], (list, tuple)):
            return False
        
        # Validate coefficients field
        if not isinstance(data["coefficients"], list):
            return False
        
        # Validate each coefficient has required fields
        for coeff in data["coefficients"]:
            if not isinstance(coeff, dict):
                return False
            
            if not self.REQUIRED_COEFFICIENT_FIELDS.issubset(coeff.keys()):
                return False
            
            # Basic type validation
            try:
                int(coeff["freq"])
                float(coeff["amp"])
                float(coeff["phase"])
            except (ValueError, TypeError):
                return False
        
        return True
