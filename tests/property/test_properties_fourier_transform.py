"""
Property-based tests for Fourier Transform Engine.

Feature: fourier-image-encryption
Property 2: DFT/IDFT Round-Trip
Property 4: Amplitude Sorting Invariant
Property 23: Coefficient Distribution Validation

These tests verify the mathematical correctness of the Fourier transform
operations and coefficient handling.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume

from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.models.data_models import FourierCoefficient


# Property 2: DFT/IDFT Round-Trip
@given(
    num_points=st.integers(min_value=10, max_value=500),
)
@pytest.mark.property_test
def test_dft_idft_round_trip(num_points):
    """
    Feature: fourier-image-encryption
    Property 2: DFT/IDFT Round-Trip
    
    For any sequence of complex numbers representing contour points,
    computing the DFT and then the IDFT should reconstruct the original
    sequence within numerical precision (ε < 1e-10).
    
    **Validates: Requirements 3.2.1, 3.2.6**
    """
    # Generate random complex contour points
    real_parts = np.random.uniform(-1000, 1000, num_points)
    imag_parts = np.random.uniform(-1000, 1000, num_points)
    original_points = real_parts + 1j * imag_parts
    
    # Create transformer
    transformer = FourierTransformer()
    
    # Compute DFT
    coefficients = transformer.compute_dft(original_points)
    
    # Verify we got the correct number of coefficients
    assert len(coefficients) == num_points
    
    # Compute IDFT to reconstruct
    reconstructed_points = transformer.compute_idft(coefficients)
    
    # Verify reconstruction matches original within numerical precision
    assert np.allclose(original_points, reconstructed_points, rtol=1e-10, atol=1e-10)
    
    # Verify shape is preserved
    assert reconstructed_points.shape == original_points.shape


@given(
    real_values=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200
    ),
    imag_values=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200
    ),
)
@pytest.mark.property_test
def test_dft_idft_round_trip_arbitrary_points(real_values, imag_values):
    """
    Feature: fourier-image-encryption
    Property 2: DFT/IDFT Round-Trip (arbitrary points)
    
    For any arbitrary sequence of complex numbers, the DFT/IDFT round-trip
    should preserve the original data.
    
    **Validates: Requirements 3.2.1, 3.2.6**
    """
    # Ensure both lists have the same length
    min_len = min(len(real_values), len(imag_values))
    assume(min_len >= 10)
    
    real_values = real_values[:min_len]
    imag_values = imag_values[:min_len]
    
    # Create complex points
    original_points = np.array([r + 1j * i for r, i in zip(real_values, imag_values)])
    
    transformer = FourierTransformer()
    
    # DFT -> IDFT round-trip
    coefficients = transformer.compute_dft(original_points)
    reconstructed_points = transformer.compute_idft(coefficients)
    
    # Verify round-trip preserves data
    assert np.allclose(original_points, reconstructed_points, rtol=1e-10, atol=1e-10)


@given(
    num_points=st.integers(min_value=10, max_value=100),
)
@pytest.mark.property_test
def test_dft_idft_round_trip_integer_coordinates(num_points):
    """
    Feature: fourier-image-encryption
    Property 2: DFT/IDFT Round-Trip (integer coordinates)
    
    For integer coordinates (typical in image processing), the round-trip
    should be highly accurate.
    
    **Validates: Requirements 3.2.1, 3.2.6**
    """
    # Generate integer coordinates (typical for image pixels)
    x_coords = np.random.randint(-500, 500, num_points)
    y_coords = np.random.randint(-500, 500, num_points)
    original_points = x_coords + 1j * y_coords
    
    transformer = FourierTransformer()
    
    coefficients = transformer.compute_dft(original_points)
    reconstructed_points = transformer.compute_idft(coefficients)
    
    # For integer coordinates, expect very high precision
    assert np.allclose(original_points, reconstructed_points, rtol=1e-12, atol=1e-12)


@pytest.mark.property_test
def test_dft_idft_round_trip_single_frequency():
    """
    Feature: fourier-image-encryption
    Property 2: DFT/IDFT Round-Trip (single frequency)
    
    For a pure sinusoidal signal (single frequency), the DFT/IDFT round-trip
    should be exact.
    
    **Validates: Requirements 3.2.1, 3.2.6**
    """
    # Create a pure sine wave in complex form
    num_points = 64
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    frequency = 5
    original_points = np.exp(1j * frequency * t)
    
    transformer = FourierTransformer()
    
    coefficients = transformer.compute_dft(original_points)
    reconstructed_points = transformer.compute_idft(coefficients)
    
    # Should be exact for pure frequency
    assert np.allclose(original_points, reconstructed_points, rtol=1e-12, atol=1e-12)


# Property 4: Amplitude Sorting Invariant
@given(
    num_points=st.integers(min_value=10, max_value=200),
)
@pytest.mark.property_test
def test_amplitude_sorting_invariant(num_points):
    """
    Feature: fourier-image-encryption
    Property 4: Amplitude Sorting Invariant
    
    For any list of Fourier coefficients after sorting by amplitude,
    each coefficient's amplitude must be greater than or equal to the
    next coefficient's amplitude (monotonically decreasing).
    
    **Validates: Requirements 3.2.3**
    """
    # Generate random complex points
    real_parts = np.random.uniform(-100, 100, num_points)
    imag_parts = np.random.uniform(-100, 100, num_points)
    points = real_parts + 1j * imag_parts
    
    transformer = FourierTransformer()
    
    # Compute DFT
    coefficients = transformer.compute_dft(points)
    
    # Sort by amplitude
    sorted_coefficients = transformer.sort_by_amplitude(coefficients)
    
    # Verify monotonically decreasing amplitudes
    for i in range(len(sorted_coefficients) - 1):
        current_amplitude = sorted_coefficients[i].amplitude
        next_amplitude = sorted_coefficients[i + 1].amplitude
        
        # Current amplitude should be >= next amplitude
        assert current_amplitude >= next_amplitude, (
            f"Amplitude at index {i} ({current_amplitude}) is less than "
            f"amplitude at index {i+1} ({next_amplitude})"
        )


@given(
    num_coefficients=st.integers(min_value=10, max_value=100),
)
@pytest.mark.property_test
def test_amplitude_sorting_preserves_coefficients(num_coefficients):
    """
    Feature: fourier-image-encryption
    Property 4: Amplitude Sorting Invariant (preservation)
    
    Sorting coefficients by amplitude should preserve all coefficients,
    only changing their order.
    
    **Validates: Requirements 3.2.3**
    """
    # Generate random complex points
    points = np.random.randn(num_coefficients) + 1j * np.random.randn(num_coefficients)
    
    transformer = FourierTransformer()
    
    coefficients = transformer.compute_dft(points)
    sorted_coefficients = transformer.sort_by_amplitude(coefficients)
    
    # Verify same number of coefficients
    assert len(sorted_coefficients) == len(coefficients)
    
    # Verify all original coefficients are present (by frequency)
    original_frequencies = sorted([c.frequency for c in coefficients])
    sorted_frequencies = sorted([c.frequency for c in sorted_coefficients])
    assert original_frequencies == sorted_frequencies
    
    # Verify amplitudes are in descending order
    amplitudes = [c.amplitude for c in sorted_coefficients]
    assert amplitudes == sorted(amplitudes, reverse=True)


@given(
    amplitudes=st.lists(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    ),
)
@pytest.mark.property_test
def test_amplitude_sorting_with_known_amplitudes(amplitudes):
    """
    Feature: fourier-image-encryption
    Property 4: Amplitude Sorting Invariant (known amplitudes)
    
    When given coefficients with known amplitudes, sorting should produce
    the expected descending order.
    
    **Validates: Requirements 3.2.3**
    """
    # Create coefficients with known amplitudes
    coefficients = []
    for i, amp in enumerate(amplitudes):
        phase = np.random.uniform(-np.pi, np.pi)
        complex_val = amp * np.exp(1j * phase)
        coeff = FourierCoefficient(
            frequency=i,
            amplitude=amp,
            phase=phase,
            complex_value=complex_val
        )
        coefficients.append(coeff)
    
    transformer = FourierTransformer()
    sorted_coefficients = transformer.sort_by_amplitude(coefficients)
    
    # Verify descending order
    sorted_amplitudes = [c.amplitude for c in sorted_coefficients]
    expected_amplitudes = sorted(amplitudes, reverse=True)
    
    assert np.allclose(sorted_amplitudes, expected_amplitudes, rtol=1e-10, atol=1e-10)


@pytest.mark.property_test
def test_amplitude_sorting_with_equal_amplitudes():
    """
    Feature: fourier-image-encryption
    Property 4: Amplitude Sorting Invariant (equal amplitudes)
    
    When coefficients have equal amplitudes, sorting should handle them
    gracefully (stable sort).
    
    **Validates: Requirements 3.2.3**
    """
    # Create coefficients with equal amplitudes
    num_coefficients = 20
    amplitude = 10.0
    coefficients = []
    
    for i in range(num_coefficients):
        phase = np.random.uniform(-np.pi, np.pi)
        complex_val = amplitude * np.exp(1j * phase)
        coeff = FourierCoefficient(
            frequency=i,
            amplitude=amplitude,
            phase=phase,
            complex_value=complex_val
        )
        coefficients.append(coeff)
    
    transformer = FourierTransformer()
    sorted_coefficients = transformer.sort_by_amplitude(coefficients)
    
    # All amplitudes should be equal
    for coeff in sorted_coefficients:
        assert np.isclose(coeff.amplitude, amplitude, rtol=1e-10, atol=1e-10)


# Property 23: Coefficient Distribution Validation
@given(
    num_points=st.integers(min_value=50, max_value=300),
)
@pytest.mark.property_test
def test_coefficient_distribution_power_law_decay(num_points):
    """
    Feature: fourier-image-encryption
    Property 23: Coefficient Distribution Validation
    
    For any valid set of Fourier coefficients from a typical contour,
    the amplitude distribution should approximately follow a power-law decay
    (each amplitude <= previous amplitude when sorted).
    
    **Validates: Requirements 3.10.3**
    """
    # Generate a typical contour-like signal (smooth curve)
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Combine a few low frequencies to create a smooth contour
    signal = (
        10 * np.exp(1j * t) +
        5 * np.exp(1j * 2 * t) +
        2 * np.exp(1j * 3 * t) +
        1 * np.exp(1j * 4 * t)
    )
    
    transformer = FourierTransformer()
    
    coefficients = transformer.compute_dft(signal)
    sorted_coefficients = transformer.sort_by_amplitude(coefficients)
    
    # Extract amplitudes
    amplitudes = np.array([c.amplitude for c in sorted_coefficients])
    
    # Verify power-law decay: amplitudes should be monotonically decreasing
    for i in range(len(amplitudes) - 1):
        assert amplitudes[i] >= amplitudes[i + 1], (
            f"Amplitude at index {i} ({amplitudes[i]}) is less than "
            f"amplitude at index {i+1} ({amplitudes[i+1]})"
        )
    
    # Verify that the first few coefficients dominate (power-law characteristic)
    # Top 10% of coefficients should contain significant energy
    top_10_percent = max(1, num_points // 10)
    top_energy = np.sum(amplitudes[:top_10_percent] ** 2)
    total_energy = np.sum(amplitudes ** 2)
    
    # Top coefficients should contain at least 50% of total energy
    energy_ratio = top_energy / total_energy if total_energy > 0 else 0
    assert energy_ratio >= 0.5, (
        f"Top {top_10_percent} coefficients contain only {energy_ratio:.2%} "
        f"of total energy, expected >= 50%"
    )


@given(
    num_points=st.integers(min_value=20, max_value=100),
)
@pytest.mark.property_test
def test_coefficient_distribution_sorted_order(num_points):
    """
    Feature: fourier-image-encryption
    Property 23: Coefficient Distribution Validation (sorted order)
    
    After sorting by amplitude, the coefficient distribution must be
    in descending order (no amplitude increases).
    
    **Validates: Requirements 3.10.3**
    """
    # Generate random signal
    points = np.random.randn(num_points) + 1j * np.random.randn(num_points)
    
    transformer = FourierTransformer()
    
    coefficients = transformer.compute_dft(points)
    sorted_coefficients = transformer.sort_by_amplitude(coefficients)
    
    # Extract amplitudes
    amplitudes = [c.amplitude for c in sorted_coefficients]
    
    # Verify descending order (power-law decay pattern)
    for i in range(len(amplitudes) - 1):
        assert amplitudes[i] >= amplitudes[i + 1]


@pytest.mark.property_test
def test_coefficient_distribution_dc_component_largest():
    """
    Feature: fourier-image-encryption
    Property 23: Coefficient Distribution Validation (DC component)
    
    For signals with a non-zero mean, the DC component (frequency 0)
    typically has the largest amplitude after sorting.
    
    **Validates: Requirements 3.10.3**
    """
    # Create signal with strong DC component (non-zero mean)
    num_points = 100
    dc_offset = 50.0
    signal = dc_offset + np.random.randn(num_points) + 1j * np.random.randn(num_points)
    
    transformer = FourierTransformer()
    
    coefficients = transformer.compute_dft(signal)
    sorted_coefficients = transformer.sort_by_amplitude(coefficients)
    
    # The largest amplitude should be the DC component (frequency 0)
    largest_coeff = sorted_coefficients[0]
    
    # Find the DC component in the original coefficients
    dc_coeff = next(c for c in coefficients if c.frequency == 0)
    
    # DC component should have the largest amplitude
    assert largest_coeff.frequency == dc_coeff.frequency
    assert np.isclose(largest_coeff.amplitude, dc_coeff.amplitude, rtol=1e-10)


@given(
    num_points=st.integers(min_value=30, max_value=150),
    noise_level=st.floats(min_value=0.01, max_value=1.0),
)
@pytest.mark.property_test
def test_coefficient_distribution_with_noise(num_points, noise_level):
    """
    Feature: fourier-image-encryption
    Property 23: Coefficient Distribution Validation (with noise)
    
    Even with added noise, the sorted coefficient distribution should
    maintain descending order.
    
    **Validates: Requirements 3.10.3**
    """
    # Create smooth signal with noise
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    clean_signal = 10 * np.exp(1j * t) + 5 * np.exp(1j * 2 * t)
    noise = noise_level * (np.random.randn(num_points) + 1j * np.random.randn(num_points))
    noisy_signal = clean_signal + noise
    
    transformer = FourierTransformer()
    
    coefficients = transformer.compute_dft(noisy_signal)
    sorted_coefficients = transformer.sort_by_amplitude(coefficients)
    
    # Verify descending order is maintained
    amplitudes = [c.amplitude for c in sorted_coefficients]
    for i in range(len(amplitudes) - 1):
        assert amplitudes[i] >= amplitudes[i + 1]


@pytest.mark.property_test
def test_coefficient_distribution_energy_conservation():
    """
    Feature: fourier-image-encryption
    Property 23: Coefficient Distribution Validation (energy conservation)
    
    The total energy (sum of squared amplitudes) should be conserved
    in the DFT, following Parseval's theorem.
    
    **Validates: Requirements 3.10.3**
    """
    # Generate random signal
    num_points = 100
    signal = np.random.randn(num_points) + 1j * np.random.randn(num_points)
    
    transformer = FourierTransformer()
    
    coefficients = transformer.compute_dft(signal)
    
    # Calculate energy in time domain
    time_energy = np.sum(np.abs(signal) ** 2)
    
    # Calculate energy in frequency domain
    freq_energy = np.sum([c.amplitude ** 2 for c in coefficients])
    
    # Parseval's theorem: energies should be equal (within numerical precision)
    # Note: NumPy's FFT doesn't normalize, so we need to account for that
    assert np.isclose(time_energy, freq_energy / num_points, rtol=1e-8, atol=1e-8)
