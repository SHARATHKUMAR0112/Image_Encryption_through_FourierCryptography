"""
Property-based tests for configuration management.

Feature: fourier-image-encryption
Property 5: Coefficient Count Validation
Property 26: Configuration Completeness

These tests verify that configuration validation correctly accepts valid values
and rejects invalid values, and that all required configuration sections are present.
"""

import json
import math
import pytest
from hypothesis import given, strategies as st
from pathlib import Path
from tempfile import NamedTemporaryFile

from fourier_encryption.config import (
    EncryptionConfig,
    PreprocessConfig,
    SystemConfig,
)
from fourier_encryption.models.exceptions import ConfigurationError


# Property 5: Coefficient Count Validation
@given(
    num_coefficients=st.integers(min_value=10, max_value=1000)
)
@pytest.mark.property_test
def test_valid_coefficient_count_accepted(num_coefficients):
    """
    Feature: fourier-image-encryption
    Property 5: Coefficient Count Validation
    
    For any requested number of Fourier terms N in the valid range [10, 1000],
    the system should accept it without raising a validation error.
    
    **Validates: Requirements 3.2.4**
    """
    config = SystemConfig()
    config.encryption.num_coefficients = num_coefficients
    config.encryption.use_ai_edge_detection = False
    config.encryption.use_ai_optimization = False
    config.encryption.use_anomaly_detection = False
    
    # Should not raise any exception
    config.validate()
    
    # Verify the value was set correctly
    assert config.encryption.num_coefficients == num_coefficients


@given(
    num_coefficients=st.one_of(
        st.integers(min_value=-1000, max_value=9),
        st.integers(min_value=1001, max_value=10000)
    )
)
@pytest.mark.property_test
def test_invalid_coefficient_count_rejected(num_coefficients):
    """
    Feature: fourier-image-encryption
    Property 5: Coefficient Count Validation (negative case)
    
    For any requested number of Fourier terms N outside the valid range [10, 1000],
    the system should reject it with a ConfigurationError.
    
    **Validates: Requirements 3.2.4**
    """
    config = SystemConfig()
    config.encryption.num_coefficients = num_coefficients
    config.encryption.use_ai_edge_detection = False
    config.encryption.use_ai_optimization = False
    config.encryption.use_anomaly_detection = False
    
    with pytest.raises(ConfigurationError) as exc_info:
        config.validate()
    
    assert "num_coefficients must be between 10 and 1000" in str(exc_info.value)


@given(
    animation_speed=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
)
@pytest.mark.property_test
def test_valid_animation_speed_accepted(animation_speed):
    """
    Feature: fourier-image-encryption
    Property 5: Coefficient Count Validation (extended to animation speed)
    
    For any animation speed in the valid range [0.1, 10.0],
    the system should accept it without raising a validation error.
    
    **Validates: Requirements 3.14.3**
    """
    config = SystemConfig()
    config.encryption.animation_speed = animation_speed
    config.encryption.use_ai_edge_detection = False
    config.encryption.use_ai_optimization = False
    config.encryption.use_anomaly_detection = False
    
    # Should not raise any exception
    config.validate()
    
    # Verify the value was set correctly
    assert abs(config.encryption.animation_speed - animation_speed) < 1e-9


@given(
    animation_speed=st.one_of(
        st.floats(min_value=-100.0, max_value=0.09, allow_nan=False, allow_infinity=False),
        st.floats(min_value=10.01, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
)
@pytest.mark.property_test
def test_invalid_animation_speed_rejected(animation_speed):
    """
    Feature: fourier-image-encryption
    Property 5: Coefficient Count Validation (extended to animation speed)
    
    For any animation speed outside the valid range [0.1, 10.0],
    the system should reject it with a ConfigurationError.
    
    **Validates: Requirements 3.14.3**
    """
    config = SystemConfig()
    config.encryption.animation_speed = animation_speed
    config.encryption.use_ai_edge_detection = False
    config.encryption.use_ai_optimization = False
    config.encryption.use_anomaly_detection = False
    
    with pytest.raises(ConfigurationError) as exc_info:
        config.validate()
    
    assert "animation_speed must be between 0.1 and 10.0" in str(exc_info.value)


@given(
    kdf_iterations=st.integers(min_value=100_000, max_value=1_000_000)
)
@pytest.mark.property_test
def test_valid_kdf_iterations_accepted(kdf_iterations):
    """
    Feature: fourier-image-encryption
    Property 5: Coefficient Count Validation (extended to KDF iterations)
    
    For any KDF iterations >= 100,000, the system should accept it.
    
    **Validates: Requirements 3.14.3**
    """
    config = SystemConfig()
    config.encryption.kdf_iterations = kdf_iterations
    config.encryption.use_ai_edge_detection = False
    config.encryption.use_ai_optimization = False
    config.encryption.use_anomaly_detection = False
    
    # Should not raise any exception
    config.validate()
    
    assert config.encryption.kdf_iterations == kdf_iterations


@given(
    kdf_iterations=st.integers(min_value=1, max_value=99_999)
)
@pytest.mark.property_test
def test_invalid_kdf_iterations_rejected(kdf_iterations):
    """
    Feature: fourier-image-encryption
    Property 5: Coefficient Count Validation (extended to KDF iterations)
    
    For any KDF iterations < 100,000, the system should reject it.
    
    **Validates: Requirements 3.14.3**
    """
    config = SystemConfig()
    config.encryption.kdf_iterations = kdf_iterations
    config.encryption.use_ai_edge_detection = False
    config.encryption.use_ai_optimization = False
    config.encryption.use_anomaly_detection = False
    
    with pytest.raises(ConfigurationError) as exc_info:
        config.validate()
    
    assert "kdf_iterations must be at least 100,000" in str(exc_info.value)


# Property 26: Configuration Completeness
@given(
    num_coefficients=st.one_of(st.none(), st.integers(min_value=10, max_value=1000)),
    use_ai_edge_detection=st.booleans(),
    use_ai_optimization=st.booleans(),
    use_anomaly_detection=st.booleans(),
    kdf_iterations=st.integers(min_value=100_000, max_value=500_000),
    animation_speed=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    target_width=st.integers(min_value=100, max_value=4000),
    target_height=st.integers(min_value=100, max_value=4000),
    denoise_strength=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@pytest.mark.property_test
def test_configuration_completeness_all_sections_present(
    num_coefficients,
    use_ai_edge_detection,
    use_ai_optimization,
    use_anomaly_detection,
    kdf_iterations,
    animation_speed,
    target_width,
    target_height,
    denoise_strength,
):
    """
    Feature: fourier-image-encryption
    Property 26: Configuration Completeness
    
    For any loaded configuration with all required sections (encryption, preprocessing,
    ai_models, performance, logging), the system should accept it if all values are valid.
    
    **Validates: Requirements 3.14.2, 3.14.3**
    """
    # Create a complete configuration
    config = SystemConfig(
        encryption=EncryptionConfig(
            num_coefficients=num_coefficients,
            use_ai_edge_detection=use_ai_edge_detection,
            use_ai_optimization=use_ai_optimization,
            use_anomaly_detection=use_anomaly_detection,
            kdf_iterations=kdf_iterations,
            animation_speed=animation_speed,
        ),
        preprocessing=PreprocessConfig(
            target_size=(target_width, target_height),
            denoise_strength=denoise_strength,
        ),
        ai_models={},
        performance={"threads": 4, "gpu_enabled": False},
        logging={"level": "INFO", "output": "stdout"},
    )
    
    # Disable AI features to avoid model path validation
    config.encryption.use_ai_edge_detection = False
    config.encryption.use_ai_optimization = False
    config.encryption.use_anomaly_detection = False
    
    # Should not raise any exception - all sections are present
    config.validate()
    
    # Verify all sections are present and accessible
    assert config.encryption is not None
    assert config.preprocessing is not None
    assert config.ai_models is not None
    assert config.performance is not None
    assert config.logging is not None
    
    # Verify encryption section has all required fields
    assert hasattr(config.encryption, 'num_coefficients')
    assert hasattr(config.encryption, 'use_ai_edge_detection')
    assert hasattr(config.encryption, 'use_ai_optimization')
    assert hasattr(config.encryption, 'use_anomaly_detection')
    assert hasattr(config.encryption, 'kdf_iterations')
    assert hasattr(config.encryption, 'animation_speed')
    
    # Verify preprocessing section has all required fields
    assert hasattr(config.preprocessing, 'target_size')
    assert hasattr(config.preprocessing, 'maintain_aspect_ratio')
    assert hasattr(config.preprocessing, 'normalize')
    assert hasattr(config.preprocessing, 'denoise')
    assert hasattr(config.preprocessing, 'denoise_strength')


@given(
    num_coefficients=st.one_of(st.none(), st.integers(min_value=10, max_value=1000)),
    kdf_iterations=st.integers(min_value=100_000, max_value=500_000),
    animation_speed=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@pytest.mark.property_test
def test_configuration_completeness_from_file(
    num_coefficients,
    kdf_iterations,
    animation_speed,
):
    """
    Feature: fourier-image-encryption
    Property 26: Configuration Completeness
    
    For any configuration loaded from a file with all required sections present,
    the system should successfully load and validate it.
    
    **Validates: Requirements 3.14.2, 3.14.3**
    """
    config_data = {
        "encryption": {
            "num_coefficients": num_coefficients,
            "use_ai_edge_detection": False,
            "use_ai_optimization": False,
            "use_anomaly_detection": False,
            "kdf_iterations": kdf_iterations,
            "animation_speed": animation_speed,
        },
        "preprocessing": {
            "target_size": [1920, 1080],
            "maintain_aspect_ratio": True,
            "normalize": True,
            "denoise": False,
            "denoise_strength": 0.5,
        },
        "ai_models": {},
        "performance": {"threads": 4},
        "logging": {"level": "INFO"},
    }
    
    with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = Path(f.name)
    
    try:
        # Load configuration from file
        config = SystemConfig.from_file(temp_path)
        
        # Verify all sections are present
        assert config.encryption is not None
        assert config.preprocessing is not None
        assert config.ai_models is not None
        assert config.performance is not None
        assert config.logging is not None
        
        # Verify values were loaded correctly
        assert config.encryption.num_coefficients == num_coefficients
        assert config.encryption.kdf_iterations == kdf_iterations
        assert abs(config.encryption.animation_speed - animation_speed) < 1e-9
        assert config.preprocessing.target_size == (1920, 1080)
        assert config.performance["threads"] == 4
        assert config.logging["level"] == "INFO"
        
    finally:
        temp_path.unlink()


@pytest.mark.property_test
def test_configuration_completeness_missing_section_uses_defaults():
    """
    Feature: fourier-image-encryption
    Property 26: Configuration Completeness
    
    When a configuration file is missing optional sections, the system should
    use default values for those sections.
    
    **Validates: Requirements 3.14.2**
    """
    # Minimal configuration with only encryption section
    config_data = {
        "encryption": {
            "num_coefficients": 100,
            "use_ai_edge_detection": False,
            "use_ai_optimization": False,
            "use_anomaly_detection": False,
        }
    }
    
    with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = Path(f.name)
    
    try:
        config = SystemConfig.from_file(temp_path)
        
        # All sections should be present (using defaults where not specified)
        assert config.encryption is not None
        assert config.preprocessing is not None
        assert config.ai_models is not None
        assert config.performance is not None
        assert config.logging is not None
        
        # Verify defaults are used for missing sections
        assert config.preprocessing.target_size == (1920, 1080)  # default
        assert config.ai_models == {}  # default
        assert config.performance == {}  # default
        assert config.logging == {}  # default
        
    finally:
        temp_path.unlink()


@given(
    denoise_strength=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@pytest.mark.property_test
def test_valid_denoise_strength_accepted(denoise_strength):
    """
    Feature: fourier-image-encryption
    Property 5: Coefficient Count Validation (extended to denoise strength)
    
    For any denoise strength in the valid range [0.0, 1.0],
    the system should accept it without raising a validation error.
    
    **Validates: Requirements 3.14.3**
    """
    config = SystemConfig()
    config.preprocessing.denoise_strength = denoise_strength
    config.encryption.use_ai_edge_detection = False
    config.encryption.use_ai_optimization = False
    config.encryption.use_anomaly_detection = False
    
    # Should not raise any exception
    config.validate()
    
    assert abs(config.preprocessing.denoise_strength - denoise_strength) < 1e-9


@given(
    denoise_strength=st.one_of(
        st.floats(min_value=-10.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
)
@pytest.mark.property_test
def test_invalid_denoise_strength_rejected(denoise_strength):
    """
    Feature: fourier-image-encryption
    Property 5: Coefficient Count Validation (extended to denoise strength)
    
    For any denoise strength outside the valid range [0.0, 1.0],
    the system should reject it with a ConfigurationError.
    
    **Validates: Requirements 3.14.3**
    """
    config = SystemConfig()
    config.preprocessing.denoise_strength = denoise_strength
    config.encryption.use_ai_edge_detection = False
    config.encryption.use_ai_optimization = False
    config.encryption.use_anomaly_detection = False
    
    with pytest.raises(ConfigurationError) as exc_info:
        config.validate()
    
    assert "denoise_strength must be between 0.0 and 1.0" in str(exc_info.value)
