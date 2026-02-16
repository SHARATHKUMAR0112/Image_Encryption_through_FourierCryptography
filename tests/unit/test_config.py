"""
Unit tests for configuration management.

Tests configuration loading, validation, and error handling.
"""

import json
import os
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from fourier_encryption.config import (
    EncryptionConfig,
    PreprocessConfig,
    SystemConfig,
)
from fourier_encryption.models.exceptions import ConfigurationError


class TestPreprocessConfig:
    """Tests for PreprocessConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = PreprocessConfig()
        assert config.target_size == (1920, 1080)
        assert config.maintain_aspect_ratio is True
        assert config.normalize is True
        assert config.denoise is False
        assert config.denoise_strength == 0.5
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = PreprocessConfig(
            target_size=(800, 600),
            maintain_aspect_ratio=False,
            normalize=False,
            denoise=True,
            denoise_strength=0.8
        )
        assert config.target_size == (800, 600)
        assert config.maintain_aspect_ratio is False
        assert config.normalize is False
        assert config.denoise is True
        assert config.denoise_strength == 0.8


class TestEncryptionConfig:
    """Tests for EncryptionConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = EncryptionConfig()
        assert config.num_coefficients is None
        assert config.use_ai_edge_detection is True
        assert config.use_ai_optimization is True
        assert config.use_anomaly_detection is True
        assert config.kdf_iterations == 100_000
        assert config.visualization_enabled is False
        assert config.animation_speed == 1.0
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = EncryptionConfig(
            num_coefficients=500,
            use_ai_edge_detection=False,
            kdf_iterations=200_000,
            animation_speed=2.0
        )
        assert config.num_coefficients == 500
        assert config.use_ai_edge_detection is False
        assert config.kdf_iterations == 200_000
        assert config.animation_speed == 2.0


class TestSystemConfig:
    """Tests for SystemConfig loading and validation."""
    
    def test_default_values(self):
        """Test that default SystemConfig has valid defaults."""
        config = SystemConfig()
        assert isinstance(config.encryption, EncryptionConfig)
        assert isinstance(config.preprocessing, PreprocessConfig)
        assert isinstance(config.ai_models, dict)
        assert isinstance(config.performance, dict)
        assert isinstance(config.logging, dict)
    
    def test_load_valid_json(self):
        """Test loading valid JSON configuration."""
        config_data = {
            "encryption": {
                "num_coefficients": 100,
                "use_ai_edge_detection": False,
                "use_ai_optimization": False,
                "use_anomaly_detection": False,
                "kdf_iterations": 150_000,
                "animation_speed": 1.5
            },
            "preprocessing": {
                "target_size": [800, 600],
                "denoise": True
            },
            "ai_models": {},
            "performance": {"threads": 4},
            "logging": {"level": "INFO"}
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = SystemConfig.from_file(temp_path)
            assert config.encryption.num_coefficients == 100
            assert config.encryption.use_ai_edge_detection is False
            assert config.encryption.use_ai_optimization is False
            assert config.encryption.use_anomaly_detection is False
            assert config.encryption.kdf_iterations == 150_000
            assert config.encryption.animation_speed == 1.5
            assert config.preprocessing.target_size == (800, 600)
            assert config.preprocessing.denoise is True
            assert config.performance["threads"] == 4
            assert config.logging["level"] == "INFO"
        finally:
            temp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            SystemConfig.from_file(Path("nonexistent.json"))
        assert "not found" in str(exc_info.value)
    
    def test_load_unsupported_format(self):
        """Test that unsupported file format raises ConfigurationError."""
        with NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                SystemConfig.from_file(temp_path)
            assert "Unsupported configuration file format" in str(exc_info.value)
        finally:
            temp_path.unlink()
    
    def test_validate_coefficient_count_too_low(self):
        """Test validation rejects coefficient count below minimum."""
        config = SystemConfig()
        config.encryption.num_coefficients = 5
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "num_coefficients must be between 10 and 1000" in str(exc_info.value)
    
    def test_validate_coefficient_count_too_high(self):
        """Test validation rejects coefficient count above maximum."""
        config = SystemConfig()
        config.encryption.num_coefficients = 1500
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "num_coefficients must be between 10 and 1000" in str(exc_info.value)
    
    def test_validate_animation_speed_too_low(self):
        """Test validation rejects animation speed below minimum."""
        config = SystemConfig()
        config.encryption.animation_speed = 0.05
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "animation_speed must be between 0.1 and 10.0" in str(exc_info.value)
    
    def test_validate_animation_speed_too_high(self):
        """Test validation rejects animation speed above maximum."""
        config = SystemConfig()
        config.encryption.animation_speed = 15.0
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "animation_speed must be between 0.1 and 10.0" in str(exc_info.value)
    
    def test_validate_kdf_iterations_too_low(self):
        """Test validation rejects KDF iterations below security minimum."""
        config = SystemConfig()
        config.encryption.kdf_iterations = 50_000
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "kdf_iterations must be at least 100,000" in str(exc_info.value)
    
    def test_validate_negative_target_size(self):
        """Test validation rejects negative target size."""
        config = SystemConfig()
        config.preprocessing.target_size = (-100, 200)
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "target_size dimensions must be positive" in str(exc_info.value)
    
    def test_validate_denoise_strength_out_of_range(self):
        """Test validation rejects denoise strength outside [0, 1]."""
        config = SystemConfig()
        config.preprocessing.denoise_strength = 1.5
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "denoise_strength must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_validate_valid_config(self):
        """Test that valid configuration passes validation."""
        config = SystemConfig()
        config.encryption.num_coefficients = 100
        config.encryption.animation_speed = 2.0
        config.encryption.kdf_iterations = 150_000
        config.encryption.use_ai_edge_detection = False
        config.encryption.use_ai_optimization = False
        config.encryption.use_anomaly_detection = False
        
        # Should not raise any exception
        config.validate()
    
    def test_load_valid_yaml(self):
        """Test loading valid YAML configuration."""
        pytest.importorskip("yaml", reason="PyYAML not installed")
        
        config_yaml = """
encryption:
  num_coefficients: 200
  use_ai_edge_detection: false
  use_ai_optimization: false
  use_anomaly_detection: false
  kdf_iterations: 120000
  animation_speed: 2.5
preprocessing:
  target_size: [1024, 768]
  denoise: true
  denoise_strength: 0.7
ai_models: {}
performance:
  threads: 8
logging:
  level: DEBUG
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            temp_path = Path(f.name)
        
        try:
            config = SystemConfig.from_file(temp_path)
            assert config.encryption.num_coefficients == 200
            assert config.encryption.use_ai_edge_detection is False
            assert config.encryption.kdf_iterations == 120_000
            assert config.encryption.animation_speed == 2.5
            assert config.preprocessing.target_size == (1024, 768)
            assert config.preprocessing.denoise is True
            assert config.preprocessing.denoise_strength == 0.7
            assert config.performance["threads"] == 8
            assert config.logging["level"] == "DEBUG"
        finally:
            temp_path.unlink()
    
    def test_load_valid_yml_extension(self):
        """Test loading YAML file with .yml extension."""
        pytest.importorskip("yaml", reason="PyYAML not installed")
        
        config_yaml = """
encryption:
  num_coefficients: 50
  use_ai_edge_detection: false
  use_ai_optimization: false
  use_anomaly_detection: false
preprocessing:
  target_size: [640, 480]
ai_models: {}
"""
        
        with NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(config_yaml)
            temp_path = Path(f.name)
        
        try:
            config = SystemConfig.from_file(temp_path)
            assert config.encryption.num_coefficients == 50
            assert config.preprocessing.target_size == (640, 480)
        finally:
            temp_path.unlink()
    
    def test_env_var_override_num_coefficients(self):
        """Test environment variable override for num_coefficients."""
        config_data = {
            "encryption": {
                "num_coefficients": 100,
                "use_ai_edge_detection": False,
                "use_ai_optimization": False,
                "use_anomaly_detection": False
            },
            "preprocessing": {},
            "ai_models": {}
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            # Set environment variable
            os.environ['FOURIER_ENCRYPTION_NUM_COEFFICIENTS'] = '250'
            config = SystemConfig.from_file(temp_path)
            assert config.encryption.num_coefficients == 250
        finally:
            # Clean up environment variable
            if 'FOURIER_ENCRYPTION_NUM_COEFFICIENTS' in os.environ:
                del os.environ['FOURIER_ENCRYPTION_NUM_COEFFICIENTS']
            temp_path.unlink()
    
    def test_env_var_override_kdf_iterations(self):
        """Test environment variable override for kdf_iterations."""
        config_data = {
            "encryption": {
                "kdf_iterations": 100_000,
                "use_ai_edge_detection": False,
                "use_ai_optimization": False,
                "use_anomaly_detection": False
            },
            "preprocessing": {},
            "ai_models": {}
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            os.environ['FOURIER_ENCRYPTION_KDF_ITERATIONS'] = '200000'
            config = SystemConfig.from_file(temp_path)
            assert config.encryption.kdf_iterations == 200_000
        finally:
            if 'FOURIER_ENCRYPTION_KDF_ITERATIONS' in os.environ:
                del os.environ['FOURIER_ENCRYPTION_KDF_ITERATIONS']
            temp_path.unlink()
    
    def test_env_var_override_animation_speed(self):
        """Test environment variable override for animation_speed."""
        config_data = {
            "encryption": {
                "animation_speed": 1.0,
                "use_ai_edge_detection": False,
                "use_ai_optimization": False,
                "use_anomaly_detection": False
            },
            "preprocessing": {},
            "ai_models": {}
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            os.environ['FOURIER_ENCRYPTION_ANIMATION_SPEED'] = '3.5'
            config = SystemConfig.from_file(temp_path)
            assert config.encryption.animation_speed == 3.5
        finally:
            if 'FOURIER_ENCRYPTION_ANIMATION_SPEED' in os.environ:
                del os.environ['FOURIER_ENCRYPTION_ANIMATION_SPEED']
            temp_path.unlink()
    
    def test_env_var_override_target_size(self):
        """Test environment variable override for target_size."""
        config_data = {
            "encryption": {
                "use_ai_edge_detection": False,
                "use_ai_optimization": False,
                "use_anomaly_detection": False
            },
            "preprocessing": {
                "target_size": [1920, 1080]
            },
            "ai_models": {}
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            os.environ['FOURIER_PREPROCESSING_TARGET_SIZE'] = '800,600'
            config = SystemConfig.from_file(temp_path)
            assert config.preprocessing.target_size == (800, 600)
        finally:
            if 'FOURIER_PREPROCESSING_TARGET_SIZE' in os.environ:
                del os.environ['FOURIER_PREPROCESSING_TARGET_SIZE']
            temp_path.unlink()
    
    def test_env_var_override_denoise_strength(self):
        """Test environment variable override for denoise_strength."""
        config_data = {
            "encryption": {
                "use_ai_edge_detection": False,
                "use_ai_optimization": False,
                "use_anomaly_detection": False
            },
            "preprocessing": {
                "denoise_strength": 0.5
            },
            "ai_models": {}
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            os.environ['FOURIER_PREPROCESSING_DENOISE_STRENGTH'] = '0.9'
            config = SystemConfig.from_file(temp_path)
            assert config.preprocessing.denoise_strength == 0.9
        finally:
            if 'FOURIER_PREPROCESSING_DENOISE_STRENGTH' in os.environ:
                del os.environ['FOURIER_PREPROCESSING_DENOISE_STRENGTH']
            temp_path.unlink()
    
    def test_load_invalid_json(self):
        """Test that invalid JSON raises ConfigurationError."""
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{invalid json content")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                SystemConfig.from_file(temp_path)
            assert "Invalid JSON format" in str(exc_info.value)
        finally:
            temp_path.unlink()
    
    def test_load_missing_required_sections(self):
        """Test that configuration with missing sections uses defaults."""
        config_data = {
            "encryption": {
                "use_ai_edge_detection": False,
                "use_ai_optimization": False,
                "use_anomaly_detection": False
            }
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = SystemConfig.from_file(temp_path)
            # Should use default values for missing sections
            assert config.preprocessing.target_size == (1920, 1080)
            assert config.ai_models == {}
            assert config.performance == {}
            assert config.logging == {}
        finally:
            temp_path.unlink()
    
    def test_validate_ai_edge_detection_missing_model(self):
        """Test validation fails when AI edge detection enabled but model path missing."""
        config = SystemConfig()
        config.encryption.use_ai_edge_detection = True
        config.ai_models = {}
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "edge_detector" in str(exc_info.value)
    
    def test_validate_ai_optimization_missing_model(self):
        """Test validation fails when AI optimization enabled but model path missing."""
        config = SystemConfig()
        config.encryption.use_ai_edge_detection = False
        config.encryption.use_ai_optimization = True
        config.ai_models = {}
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "coefficient_optimizer" in str(exc_info.value)
    
    def test_validate_anomaly_detection_missing_model(self):
        """Test validation fails when anomaly detection enabled but model path missing."""
        config = SystemConfig()
        config.encryption.use_ai_edge_detection = False
        config.encryption.use_ai_optimization = False
        config.encryption.use_anomaly_detection = True
        config.ai_models = {}
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "anomaly_detector" in str(exc_info.value)
    
    def test_validate_ai_model_file_not_found(self):
        """Test validation fails when AI model file doesn't exist."""
        config = SystemConfig()
        config.encryption.use_ai_edge_detection = True
        config.ai_models = {'edge_detector': Path('/nonexistent/model.pt')}
        
        with pytest.raises(ConfigurationError) as exc_info:
            config.validate()
        assert "model file not found" in str(exc_info.value).lower()
    
    def test_load_empty_json(self):
        """Test loading empty JSON file uses all defaults."""
        config_data = {
            "encryption": {
                "use_ai_edge_detection": False,
                "use_ai_optimization": False,
                "use_anomaly_detection": False
            },
            "ai_models": {}
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = SystemConfig.from_file(temp_path)
            # Should use all default values for unspecified fields
            assert config.encryption.num_coefficients is None
            assert config.encryption.kdf_iterations == 100_000
            assert config.preprocessing.target_size == (1920, 1080)
        finally:
            temp_path.unlink()
