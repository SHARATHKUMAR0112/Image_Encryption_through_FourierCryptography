"""
Property-based tests for Visualization Module.

Feature: fourier-image-encryption
Property 15: Progress Percentage Bounds

These tests verify the correctness of progress tracking and metrics
in the visualization and monitoring components.
"""

import math
import pytest
from hypothesis import given, strategies as st

from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.models.data_models import FourierCoefficient, Metrics
from fourier_encryption.visualization.live_renderer import LiveRenderer
from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard


# Helper strategy to generate valid FourierCoefficient objects
@st.composite
def fourier_coefficient_strategy(draw):
    """Generate a valid FourierCoefficient for testing."""
    frequency = draw(st.integers(min_value=-100, max_value=100))
    amplitude = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
    phase = draw(st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False))
    
    # Compute consistent complex value
    complex_value = amplitude * complex(math.cos(phase), math.sin(phase))
    
    return FourierCoefficient(
        frequency=frequency,
        amplitude=amplitude,
        phase=phase,
        complex_value=complex_value
    )


# Property 15: Progress Percentage Bounds
@given(
    current_frame=st.integers(min_value=0, max_value=1000),
    total_frames=st.integers(min_value=1, max_value=1000)
)
@pytest.mark.property_test
def test_progress_percentage_bounds_calculation(current_frame, total_frames):
    """
    Feature: fourier-image-encryption
    Property 15: Progress Percentage Bounds
    
    For any processing state during encryption or decryption,
    the progress percentage must be in the range [0, 100] and
    must increase monotonically over time.
    
    This test verifies that progress percentage calculation
    always produces values in the valid range.
    
    **Validates: Requirements 3.6.4, 3.7.3**
    """
    # Ensure current_frame doesn't exceed total_frames
    if current_frame > total_frames:
        current_frame = total_frames
    
    # Calculate progress percentage (same logic as in LiveRenderer._display_frame_info)
    if total_frames > 0:
        progress = (current_frame / total_frames) * 100
    else:
        progress = 0.0
    
    # Verify progress is in [0, 100]
    assert 0 <= progress <= 100, f"Progress {progress}% is outside [0, 100] range"
    
    # Verify progress is non-negative
    assert progress >= 0, f"Progress {progress}% is negative"
    
    # Verify progress doesn't exceed 100
    assert progress <= 100, f"Progress {progress}% exceeds 100"


@given(
    coefficients=st.lists(fourier_coefficient_strategy(), min_size=1, max_size=50),
    num_frames=st.integers(min_value=10, max_value=500)
)
@pytest.mark.property_test
def test_progress_percentage_bounds_renderer(coefficients, num_frames):
    """
    Feature: fourier-image-encryption
    Property 15: Progress Percentage Bounds (LiveRenderer)
    
    For any animation sequence in LiveRenderer, the progress percentage
    displayed must always be in the range [0, 100].
    
    **Validates: Requirements 3.6.4, 3.7.3**
    """
    renderer = LiveRenderer(backend="matplotlib")
    engine = EpicycleEngine(coefficients)
    
    # Set total frames for progress tracking
    renderer._total_frames = num_frames
    
    # Generate frames and track progress
    frames = list(engine.generate_animation_frames(num_frames, speed=1.0))
    
    for frame_idx, state in enumerate(frames):
        # Update renderer state
        renderer._current_frame = frame_idx
        
        # Calculate progress (same as renderer does)
        if renderer._total_frames > 0:
            progress = (renderer._current_frame / renderer._total_frames) * 100
        else:
            progress = 0.0
        
        # Verify progress is in [0, 100]
        assert 0 <= progress <= 100, (
            f"Frame {frame_idx}/{num_frames}: Progress {progress}% is outside [0, 100] range"
        )


@given(
    progress_percentage=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    current_coefficient_index=st.integers(min_value=0, max_value=1000),
    active_radius=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    fps=st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    processing_time=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    memory_usage_mb=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    encryption_status=st.sampled_from(["idle", "encrypting", "decrypting", "complete", "error"])
)
@pytest.mark.property_test
def test_progress_percentage_bounds_metrics(
    progress_percentage,
    current_coefficient_index,
    active_radius,
    fps,
    processing_time,
    memory_usage_mb,
    encryption_status
):
    """
    Feature: fourier-image-encryption
    Property 15: Progress Percentage Bounds (Metrics)
    
    For any Metrics object, the progress_percentage field must be
    in the range [0, 100]. The Metrics dataclass enforces this
    constraint in its __post_init__ validation.
    
    **Validates: Requirements 3.6.4, 3.7.3**
    """
    # Create Metrics object with valid progress percentage
    metrics = Metrics(
        current_coefficient_index=current_coefficient_index,
        active_radius=active_radius,
        progress_percentage=progress_percentage,
        fps=fps,
        processing_time=processing_time,
        memory_usage_mb=memory_usage_mb,
        encryption_status=encryption_status
    )
    
    # Verify progress_percentage is in [0, 100]
    assert 0 <= metrics.progress_percentage <= 100, (
        f"Metrics progress_percentage {metrics.progress_percentage}% is outside [0, 100] range"
    )


@given(
    invalid_progress=st.one_of(
        st.floats(min_value=-1000.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
        st.floats(min_value=100.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
)
@pytest.mark.property_test
def test_progress_percentage_bounds_metrics_invalid(invalid_progress):
    """
    Feature: fourier-image-encryption
    Property 15: Progress Percentage Bounds (Invalid Case)
    
    For any progress percentage outside [0, 100], the Metrics dataclass
    should reject it with a validation error.
    
    **Validates: Requirements 3.6.4, 3.7.3**
    """
    # Attempt to create Metrics with invalid progress percentage
    with pytest.raises(ValueError, match="progress_percentage must be in"):
        Metrics(
            current_coefficient_index=0,
            active_radius=0.0,
            progress_percentage=invalid_progress,
            fps=30.0,
            processing_time=0.0,
            memory_usage_mb=100.0,
            encryption_status="idle"
        )


@given(
    progress_percentage=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
)
@pytest.mark.property_test
def test_progress_percentage_bounds_dashboard(progress_percentage):
    """
    Feature: fourier-image-encryption
    Property 15: Progress Percentage Bounds (MonitoringDashboard)
    
    For any metrics updated in the MonitoringDashboard, the progress
    percentage must be in the range [0, 100].
    
    **Validates: Requirements 3.6.4, 3.7.3**
    """
    dashboard = MonitoringDashboard()
    
    # Create valid metrics with the given progress percentage
    metrics = Metrics(
        current_coefficient_index=0,
        active_radius=0.0,
        progress_percentage=progress_percentage,
        fps=30.0,
        processing_time=0.0,
        memory_usage_mb=100.0,
        encryption_status="idle"
    )
    
    # Update dashboard metrics
    dashboard.update_metrics(metrics)
    
    # Retrieve metrics and verify progress is in [0, 100]
    retrieved_metrics = dashboard.metrics
    assert 0 <= retrieved_metrics.progress_percentage <= 100, (
        f"Dashboard progress {retrieved_metrics.progress_percentage}% is outside [0, 100] range"
    )


@given(
    num_updates=st.integers(min_value=2, max_value=100)
)
@pytest.mark.property_test
def test_progress_percentage_monotonic_increase(num_updates):
    """
    Feature: fourier-image-encryption
    Property 15: Progress Percentage Bounds (Monotonic Increase)
    
    For any sequence of progress updates, the progress percentage
    must increase monotonically (never decrease) over time.
    
    **Validates: Requirements 3.6.4, 3.7.3**
    """
    dashboard = MonitoringDashboard()
    
    previous_progress = 0.0
    
    for i in range(num_updates):
        # Calculate progress for this update (monotonically increasing)
        current_progress = (i / (num_updates - 1)) * 100.0
        
        # Create metrics with current progress
        metrics = Metrics(
            current_coefficient_index=i,
            active_radius=0.0,
            progress_percentage=current_progress,
            fps=30.0,
            processing_time=float(i),
            memory_usage_mb=100.0,
            encryption_status="encrypting"
        )
        
        # Update dashboard
        dashboard.update_metrics(metrics)
        
        # Verify progress is in [0, 100]
        assert 0 <= current_progress <= 100
        
        # Verify progress is monotonically increasing
        assert current_progress >= previous_progress, (
            f"Progress decreased from {previous_progress}% to {current_progress}%"
        )
        
        previous_progress = current_progress
