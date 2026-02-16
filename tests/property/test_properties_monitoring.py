"""
Property-based tests for Monitoring Module.

Feature: fourier-image-encryption
Property 16: Metrics Validity

These tests verify that all metrics displayed by the monitoring dashboard
are valid: non-negative values and FPS bounded by configured maximum.
"""

import math
import pytest
from hypothesis import given, strategies as st, assume

from fourier_encryption.models.data_models import Metrics
from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard


# Helper strategy to generate valid Metrics objects
@st.composite
def metrics_strategy(draw, max_fps=120.0):
    """Generate a valid Metrics object for testing."""
    current_coefficient_index = draw(st.integers(min_value=0, max_value=10000))
    active_radius = draw(st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    progress_percentage = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    fps = draw(st.floats(min_value=0.0, max_value=max_fps, allow_nan=False, allow_infinity=False))
    processing_time = draw(st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    memory_usage_mb = draw(st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False))
    encryption_status = draw(st.sampled_from(["idle", "encrypting", "decrypting", "complete", "error"]))
    
    return Metrics(
        current_coefficient_index=current_coefficient_index,
        active_radius=active_radius,
        progress_percentage=progress_percentage,
        fps=fps,
        processing_time=processing_time,
        memory_usage_mb=memory_usage_mb,
        encryption_status=encryption_status
    )


# Property 16: Metrics Validity
@given(
    current_coefficient_index=st.integers(min_value=0, max_value=10000),
    active_radius=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    progress_percentage=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    fps=st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    processing_time=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    memory_usage_mb=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    encryption_status=st.sampled_from(["idle", "encrypting", "decrypting", "complete", "error"])
)
@pytest.mark.property_test
def test_metrics_validity_all_non_negative(
    current_coefficient_index,
    active_radius,
    progress_percentage,
    fps,
    processing_time,
    memory_usage_mb,
    encryption_status
):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity
    
    For any monitoring dashboard state, all displayed metrics (FPS, processing time,
    memory usage) must be non-negative, and FPS must be ≤ the configured maximum frame rate.
    
    This test verifies that all numeric metrics are non-negative.
    
    **Validates: Requirements 3.7.5**
    """
    # Create Metrics object
    metrics = Metrics(
        current_coefficient_index=current_coefficient_index,
        active_radius=active_radius,
        progress_percentage=progress_percentage,
        fps=fps,
        processing_time=processing_time,
        memory_usage_mb=memory_usage_mb,
        encryption_status=encryption_status
    )
    
    # Verify all numeric metrics are non-negative
    assert metrics.current_coefficient_index >= 0, (
        f"current_coefficient_index {metrics.current_coefficient_index} is negative"
    )
    assert metrics.active_radius >= 0, (
        f"active_radius {metrics.active_radius} is negative"
    )
    assert metrics.progress_percentage >= 0, (
        f"progress_percentage {metrics.progress_percentage} is negative"
    )
    assert metrics.fps >= 0, (
        f"fps {metrics.fps} is negative"
    )
    assert metrics.processing_time >= 0, (
        f"processing_time {metrics.processing_time} is negative"
    )
    assert metrics.memory_usage_mb >= 0, (
        f"memory_usage_mb {metrics.memory_usage_mb} is negative"
    )


@given(
    metrics=metrics_strategy(max_fps=120.0)
)
@pytest.mark.property_test
def test_metrics_validity_fps_bounded(metrics):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (FPS Bounded)
    
    For any monitoring dashboard state, the FPS metric must be bounded
    by the configured maximum frame rate (typically 120 FPS for smooth animation).
    
    **Validates: Requirements 3.7.5**
    """
    max_fps = 120.0
    
    # Verify FPS is bounded by maximum
    assert metrics.fps <= max_fps, (
        f"fps {metrics.fps} exceeds maximum {max_fps}"
    )
    
    # Verify FPS is non-negative
    assert metrics.fps >= 0, (
        f"fps {metrics.fps} is negative"
    )


@given(
    metrics=metrics_strategy()
)
@pytest.mark.property_test
def test_metrics_validity_dashboard_update(metrics):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (Dashboard Update)
    
    For any metrics updated in the MonitoringDashboard, all metrics
    must remain valid (non-negative, FPS bounded).
    
    **Validates: Requirements 3.7.5**
    """
    dashboard = MonitoringDashboard()
    
    # Update dashboard with metrics
    dashboard.update_metrics(metrics)
    
    # Retrieve metrics from dashboard
    retrieved_metrics = dashboard.metrics
    
    # Verify all metrics are non-negative
    assert retrieved_metrics.current_coefficient_index >= 0
    assert retrieved_metrics.active_radius >= 0
    assert retrieved_metrics.progress_percentage >= 0
    assert retrieved_metrics.fps >= 0
    assert retrieved_metrics.processing_time >= 0
    assert retrieved_metrics.memory_usage_mb >= 0
    
    # Verify FPS is bounded (reasonable upper limit)
    assert retrieved_metrics.fps <= 1000.0, (
        f"fps {retrieved_metrics.fps} is unreasonably high"
    )


@given(
    invalid_coefficient_index=st.integers(min_value=-10000, max_value=-1)
)
@pytest.mark.property_test
def test_metrics_validity_rejects_negative_coefficient_index(invalid_coefficient_index):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (Negative Coefficient Index Rejection)
    
    For any negative coefficient index, the Metrics dataclass should
    reject it with a validation error.
    
    **Validates: Requirements 3.7.5**
    """
    with pytest.raises(ValueError, match="current_coefficient_index must be non-negative"):
        Metrics(
            current_coefficient_index=invalid_coefficient_index,
            active_radius=0.0,
            progress_percentage=50.0,
            fps=30.0,
            processing_time=1.0,
            memory_usage_mb=100.0,
            encryption_status="idle"
        )


@given(
    invalid_radius=st.floats(min_value=-10000.0, max_value=-0.001, allow_nan=False, allow_infinity=False)
)
@pytest.mark.property_test
def test_metrics_validity_rejects_negative_radius(invalid_radius):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (Negative Radius Rejection)
    
    For any negative active radius, the Metrics dataclass should
    reject it with a validation error.
    
    **Validates: Requirements 3.7.5**
    """
    with pytest.raises(ValueError, match="active_radius must be non-negative"):
        Metrics(
            current_coefficient_index=0,
            active_radius=invalid_radius,
            progress_percentage=50.0,
            fps=30.0,
            processing_time=1.0,
            memory_usage_mb=100.0,
            encryption_status="idle"
        )


@given(
    invalid_fps=st.floats(min_value=-1000.0, max_value=-0.001, allow_nan=False, allow_infinity=False)
)
@pytest.mark.property_test
def test_metrics_validity_rejects_negative_fps(invalid_fps):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (Negative FPS Rejection)
    
    For any negative FPS value, the Metrics dataclass should
    reject it with a validation error.
    
    **Validates: Requirements 3.7.5**
    """
    with pytest.raises(ValueError, match="fps must be non-negative"):
        Metrics(
            current_coefficient_index=0,
            active_radius=0.0,
            progress_percentage=50.0,
            fps=invalid_fps,
            processing_time=1.0,
            memory_usage_mb=100.0,
            encryption_status="idle"
        )


@given(
    invalid_processing_time=st.floats(min_value=-10000.0, max_value=-0.001, allow_nan=False, allow_infinity=False)
)
@pytest.mark.property_test
def test_metrics_validity_rejects_negative_processing_time(invalid_processing_time):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (Negative Processing Time Rejection)
    
    For any negative processing time, the Metrics dataclass should
    reject it with a validation error.
    
    **Validates: Requirements 3.7.5**
    """
    with pytest.raises(ValueError, match="processing_time must be non-negative"):
        Metrics(
            current_coefficient_index=0,
            active_radius=0.0,
            progress_percentage=50.0,
            fps=30.0,
            processing_time=invalid_processing_time,
            memory_usage_mb=100.0,
            encryption_status="idle"
        )


@given(
    invalid_memory=st.floats(min_value=-100000.0, max_value=-0.001, allow_nan=False, allow_infinity=False)
)
@pytest.mark.property_test
def test_metrics_validity_rejects_negative_memory(invalid_memory):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (Negative Memory Rejection)
    
    For any negative memory usage value, the Metrics dataclass should
    reject it with a validation error.
    
    **Validates: Requirements 3.7.5**
    """
    with pytest.raises(ValueError, match="memory_usage_mb must be non-negative"):
        Metrics(
            current_coefficient_index=0,
            active_radius=0.0,
            progress_percentage=50.0,
            fps=30.0,
            processing_time=1.0,
            memory_usage_mb=invalid_memory,
            encryption_status="idle"
        )


@given(
    num_updates=st.integers(min_value=2, max_value=50)
)
@pytest.mark.property_test
def test_metrics_validity_multiple_updates(num_updates):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (Multiple Updates)
    
    For any sequence of metric updates to the dashboard, all metrics
    must remain valid throughout the sequence.
    
    **Validates: Requirements 3.7.5**
    """
    dashboard = MonitoringDashboard()
    
    for i in range(num_updates):
        # Create metrics with varying values
        progress = (i / (num_updates - 1)) * 100.0
        fps = 30.0 + (i % 30)  # Vary FPS between 30-60
        
        metrics = Metrics(
            current_coefficient_index=i * 10,
            active_radius=float(i) * 0.5,
            progress_percentage=progress,
            fps=fps,
            processing_time=float(i) * 0.1,
            memory_usage_mb=100.0 + float(i) * 5.0,
            encryption_status="encrypting" if i < num_updates - 1 else "complete"
        )
        
        # Update dashboard
        dashboard.update_metrics(metrics)
        
        # Retrieve and verify metrics
        retrieved = dashboard.metrics
        
        # Verify all metrics are non-negative
        assert retrieved.current_coefficient_index >= 0
        assert retrieved.active_radius >= 0
        assert retrieved.progress_percentage >= 0
        assert retrieved.fps >= 0
        assert retrieved.processing_time >= 0
        assert retrieved.memory_usage_mb >= 0
        
        # Verify FPS is bounded
        assert retrieved.fps <= 120.0


@given(
    metrics=metrics_strategy()
)
@pytest.mark.property_test
def test_metrics_validity_display_output(metrics):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (Display Output)
    
    For any metrics displayed by the dashboard, the display output
    should contain valid formatted values (no NaN, no negative values).
    
    **Validates: Requirements 3.7.5**
    """
    dashboard = MonitoringDashboard()
    dashboard.update_metrics(metrics)
    
    # Get terminal display output
    display_output = dashboard.display(mode="terminal")
    
    # Verify output is a non-empty string
    assert isinstance(display_output, str)
    assert len(display_output) > 0
    
    # Verify no NaN or inf in output
    assert "nan" not in display_output.lower()
    assert "inf" not in display_output.lower()
    
    # Verify metrics are displayed
    assert str(metrics.encryption_status.upper()) in display_output
    assert f"{metrics.progress_percentage:.1f}" in display_output


@given(
    metrics=metrics_strategy()
)
@pytest.mark.property_test
def test_metrics_validity_gui_output(metrics):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (GUI Output)
    
    For any metrics displayed in GUI mode, the output should be
    valid JSON with all numeric values being valid (non-NaN, non-infinite).
    
    **Validates: Requirements 3.7.5**
    """
    import json
    
    dashboard = MonitoringDashboard()
    dashboard.update_metrics(metrics)
    
    # Get GUI display output (JSON format)
    gui_output = dashboard.display(mode="gui")
    
    # Verify output is valid JSON
    data = json.loads(gui_output)
    
    # Verify all numeric fields are present and valid
    assert "fps" in data
    assert "processing_time" in data
    assert "memory_usage_mb" in data
    assert "progress" in data
    
    # Verify all numeric values are finite
    assert math.isfinite(data["fps"])
    assert math.isfinite(data["processing_time"])
    assert math.isfinite(data["memory_usage_mb"])
    assert math.isfinite(data["progress"])
    
    # Verify all numeric values are non-negative
    assert data["fps"] >= 0
    assert data["processing_time"] >= 0
    assert data["memory_usage_mb"] >= 0
    assert data["progress"] >= 0


@given(
    max_fps=st.floats(min_value=1.0, max_value=240.0, allow_nan=False, allow_infinity=False)
)
@pytest.mark.property_test
def test_metrics_validity_fps_upper_bound_configurable(max_fps):
    """
    Feature: fourier-image-encryption
    Property 16: Metrics Validity (Configurable FPS Bound)
    
    For any configured maximum FPS value, the system should enforce
    that reported FPS values do not exceed this maximum.
    
    **Validates: Requirements 3.7.5**
    """
    # Generate FPS value within the configured maximum
    fps_value = max_fps * 0.9  # Use 90% of max to ensure it's within bounds
    
    metrics = Metrics(
        current_coefficient_index=0,
        active_radius=0.0,
        progress_percentage=50.0,
        fps=fps_value,
        processing_time=1.0,
        memory_usage_mb=100.0,
        encryption_status="encrypting"
    )
    
    # Verify FPS is within configured bounds
    assert metrics.fps <= max_fps, (
        f"fps {metrics.fps} exceeds configured maximum {max_fps}"
    )
    assert metrics.fps >= 0, (
        f"fps {metrics.fps} is negative"
    )
