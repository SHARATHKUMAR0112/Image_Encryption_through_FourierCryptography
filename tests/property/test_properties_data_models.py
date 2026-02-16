"""
Property-based tests for core data models.

Feature: fourier-image-encryption
Property 3: Coefficient Structure Completeness

These tests verify that all Fourier coefficients have valid fields and that
validation rules are enforced correctly.
"""

import math
import pytest
from hypothesis import given, strategies as st

from fourier_encryption.models import FourierCoefficient


@given(
    frequency=st.integers(min_value=-1000, max_value=1000),
    amplitude=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    phase=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
)
@pytest.mark.property_test
def test_coefficient_structure_completeness(frequency, amplitude, phase):
    """
    Feature: fourier-image-encryption
    Property 3: Coefficient Structure Completeness
    
    For any valid Fourier coefficient, all fields must be present and valid:
    - frequency: integer
    - amplitude: non-negative float
    - phase: float in [-π, π]
    - complex_value: consistent with amplitude and phase
    
    Validates: Requirements 3.2.2
    """
    # Compute complex value from amplitude and phase
    complex_value = amplitude * complex(math.cos(phase), math.sin(phase))
    
    # Create coefficient
    coeff = FourierCoefficient(
        frequency=frequency,
        amplitude=amplitude,
        phase=phase,
        complex_value=complex_value
    )
    
    # Verify all fields are present and have correct types
    assert isinstance(coeff.frequency, int)
    assert isinstance(coeff.amplitude, float)
    assert isinstance(coeff.phase, float)
    assert isinstance(coeff.complex_value, complex)
    
    # Verify amplitude is non-negative
    assert coeff.amplitude >= 0
    
    # Verify phase is in [-π, π]
    assert -math.pi <= coeff.phase <= math.pi
    
    # Verify complex_value is consistent with amplitude and phase
    expected_complex = coeff.amplitude * complex(
        math.cos(coeff.phase), math.sin(coeff.phase)
    )
    assert abs(coeff.complex_value - expected_complex) < 1e-9


@given(
    frequency=st.integers(min_value=-1000, max_value=1000),
    amplitude=st.floats(min_value=-1000.0, max_value=-0.001, allow_nan=False, allow_infinity=False),
    phase=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
)
@pytest.mark.property_test
def test_negative_amplitude_rejected(frequency, amplitude, phase):
    """
    Feature: fourier-image-encryption
    Property 3: Coefficient Structure Completeness (negative case)
    
    For any Fourier coefficient with negative amplitude, validation must fail.
    
    Validates: Requirements 3.2.2
    """
    complex_value = amplitude * complex(math.cos(phase), math.sin(phase))
    
    with pytest.raises(ValueError, match="Amplitude must be non-negative"):
        FourierCoefficient(
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            complex_value=complex_value
        )


@given(
    frequency=st.integers(min_value=-1000, max_value=1000),
    amplitude=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    phase=st.one_of(
        st.floats(min_value=-1000.0, max_value=-math.pi - 0.001, allow_nan=False, allow_infinity=False),
        st.floats(min_value=math.pi + 0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
    ),
)
@pytest.mark.property_test
def test_invalid_phase_rejected(frequency, amplitude, phase):
    """
    Feature: fourier-image-encryption
    Property 3: Coefficient Structure Completeness (negative case)
    
    For any Fourier coefficient with phase outside [-π, π], validation must fail.
    
    Validates: Requirements 3.2.2
    """
    complex_value = amplitude * complex(math.cos(phase), math.sin(phase))
    
    with pytest.raises(ValueError, match="Phase must be in"):
        FourierCoefficient(
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            complex_value=complex_value
        )
