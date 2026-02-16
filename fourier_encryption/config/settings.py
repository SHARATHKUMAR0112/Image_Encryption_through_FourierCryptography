"""
Configuration management for Fourier-Based Image Encryption System.

This module provides dataclasses for system configuration with support for
loading from YAML and JSON files, validation, and environment-specific overrides.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fourier_encryption.models.data_models import ReconstructionConfig
from fourier_encryption.models.exceptions import ConfigurationError


@dataclass
class PreprocessConfig:
    """
    Configuration for image preprocessing operations.
    
    Attributes:
        target_size: Target dimensions (width, height) for image resizing
        maintain_aspect_ratio: Whether to preserve aspect ratio during resize
        normalize: Whether to normalize pixel values to [0, 1]
        denoise: Whether to apply denoising filter
        denoise_strength: Strength of denoising (0.0 to 1.0)
    """
    target_size: Tuple[int, int] = (1920, 1080)
    maintain_aspect_ratio: bool = True
    normalize: bool = True
    denoise: bool = False
    denoise_strength: float = 0.5


@dataclass
class EncryptionConfig:
    """
    Configuration for encryption operations and AI features.
    
    Attributes:
        num_coefficients: Number of Fourier coefficients (None = auto-optimize)
        use_ai_edge_detection: Enable AI-enhanced edge detection
        use_ai_optimization: Enable AI coefficient count optimization
        use_anomaly_detection: Enable AI anomaly detection
        kdf_iterations: Number of PBKDF2 iterations for key derivation
        visualization_enabled: Enable live epicycle animation
        animation_speed: Animation playback speed multiplier (0.1 to 10.0)
        reconstruction_enabled: Enable reconstruction after decryption
    """
    num_coefficients: Optional[int] = None
    use_ai_edge_detection: bool = True
    use_ai_optimization: bool = True
    use_anomaly_detection: bool = True
    kdf_iterations: int = 100_000
    visualization_enabled: bool = False
    animation_speed: float = 1.0
    reconstruction_enabled: bool = False


@dataclass
class SystemConfig:
    """
    Complete system configuration loaded from YAML or JSON files.
    
    Attributes:
        encryption: Encryption and AI feature configuration
        preprocessing: Image preprocessing configuration
        reconstruction: Image reconstruction configuration
        ai_models: Paths to AI model files
        performance: Performance-related settings (threads, GPU)
        logging: Logging configuration (levels, output paths)
    """
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    ai_models: Dict[str, Path] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, path: Path) -> "SystemConfig":
        """
        Load and validate configuration from YAML or JSON file.
        
        Supports environment variable overrides with the following format:
        - FOURIER_ENCRYPTION_NUM_COEFFICIENTS
        - FOURIER_ENCRYPTION_KDF_ITERATIONS
        - FOURIER_ENCRYPTION_ANIMATION_SPEED
        - FOURIER_PREPROCESSING_TARGET_SIZE (format: "width,height")
        - FOURIER_PREPROCESSING_DENOISE_STRENGTH
        
        Args:
            path: Path to configuration file (.yaml, .yml, or .json)
            
        Returns:
            SystemConfig instance with loaded configuration
            
        Raises:
            ConfigurationError: If file not found, invalid format, or validation fails
        """
        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {path}",
                context={"path": str(path)}
            )
        
        # Determine file format and load
        suffix = path.suffix.lower()
        try:
            if suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                except ImportError:
                    raise ConfigurationError(
                        "YAML support requires PyYAML library. Install with: pip install pyyaml",
                        context={"file": str(path)}
                    )
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {suffix}",
                    context={"path": str(path), "supported": [".json", ".yaml", ".yml"]}
                )
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON format: {e}",
                context={"path": str(path), "error": str(e)}
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration file: {e}",
                context={"path": str(path), "error": str(e)}
            )
        
        # Parse configuration sections
        try:
            # Parse encryption config with environment variable overrides
            encryption_data = data.get('encryption', {})
            
            # Apply environment variable overrides
            num_coefficients = encryption_data.get('num_coefficients')
            if os.getenv('FOURIER_ENCRYPTION_NUM_COEFFICIENTS'):
                num_coefficients = int(os.getenv('FOURIER_ENCRYPTION_NUM_COEFFICIENTS'))
            
            kdf_iterations = encryption_data.get('kdf_iterations', 100_000)
            if os.getenv('FOURIER_ENCRYPTION_KDF_ITERATIONS'):
                kdf_iterations = int(os.getenv('FOURIER_ENCRYPTION_KDF_ITERATIONS'))
            
            animation_speed = encryption_data.get('animation_speed', 1.0)
            if os.getenv('FOURIER_ENCRYPTION_ANIMATION_SPEED'):
                animation_speed = float(os.getenv('FOURIER_ENCRYPTION_ANIMATION_SPEED'))
            
            encryption = EncryptionConfig(
                num_coefficients=num_coefficients,
                use_ai_edge_detection=encryption_data.get('use_ai_edge_detection', True),
                use_ai_optimization=encryption_data.get('use_ai_optimization', True),
                use_anomaly_detection=encryption_data.get('use_anomaly_detection', True),
                kdf_iterations=kdf_iterations,
                visualization_enabled=encryption_data.get('visualization_enabled', False),
                animation_speed=animation_speed,
                reconstruction_enabled=encryption_data.get('reconstruction_enabled', False)
            )
            
            # Parse preprocessing config with environment variable overrides
            preprocess_data = data.get('preprocessing', {})
            target_size = preprocess_data.get('target_size', [1920, 1080])
            
            if os.getenv('FOURIER_PREPROCESSING_TARGET_SIZE'):
                size_str = os.getenv('FOURIER_PREPROCESSING_TARGET_SIZE')
                width, height = map(int, size_str.split(','))
                target_size = [width, height]
            
            denoise_strength = preprocess_data.get('denoise_strength', 0.5)
            if os.getenv('FOURIER_PREPROCESSING_DENOISE_STRENGTH'):
                denoise_strength = float(os.getenv('FOURIER_PREPROCESSING_DENOISE_STRENGTH'))
            
            preprocessing = PreprocessConfig(
                target_size=tuple(target_size) if isinstance(target_size, list) else target_size,
                maintain_aspect_ratio=preprocess_data.get('maintain_aspect_ratio', True),
                normalize=preprocess_data.get('normalize', True),
                denoise=preprocess_data.get('denoise', False),
                denoise_strength=denoise_strength
            )
            
            # Parse reconstruction config
            reconstruction_data = data.get('reconstruction', {})
            reconstruction = ReconstructionConfig(
                mode=reconstruction_data.get('mode', 'animated'),
                speed=reconstruction_data.get('speed', 1.0),
                quality=reconstruction_data.get('quality', 'balanced'),
                save_frames=reconstruction_data.get('save_frames', False),
                save_animation=reconstruction_data.get('save_animation', False),
                output_format=reconstruction_data.get('output_format', 'mp4'),
                output_path=reconstruction_data.get('output_path'),
                backend=reconstruction_data.get('backend', 'pyqtgraph')
            )
            
            # Parse AI models paths
            ai_models_data = data.get('ai_models', {})
            ai_models = {k: Path(v) for k, v in ai_models_data.items()}
            
            # Parse performance settings
            performance = data.get('performance', {})
            
            # Parse logging settings
            logging_config = data.get('logging', {})
            
            config = cls(
                encryption=encryption,
                preprocessing=preprocessing,
                reconstruction=reconstruction,
                ai_models=ai_models,
                performance=performance,
                logging=logging_config
            )
            
            # Validate the loaded configuration
            config.validate()
            
            return config
            
        except (KeyError, ValueError, TypeError) as e:
            raise ConfigurationError(
                f"Invalid configuration structure: {e}",
                context={"path": str(path), "error": str(e)}
            )
    
    def validate(self) -> None:
        """
        Validate all configuration values.
        
        Raises:
            ConfigurationError: If any configuration value is invalid
        """
        # Validate encryption config
        if self.encryption.num_coefficients is not None:
            if not (10 <= self.encryption.num_coefficients <= 1000):
                raise ConfigurationError(
                    "num_coefficients must be between 10 and 1000",
                    context={
                        "value": self.encryption.num_coefficients,
                        "valid_range": "[10, 1000]"
                    }
                )
        
        if not (0.1 <= self.encryption.animation_speed <= 10.0):
            raise ConfigurationError(
                "animation_speed must be between 0.1 and 10.0",
                context={
                    "value": self.encryption.animation_speed,
                    "valid_range": "[0.1, 10.0]"
                }
            )
        
        if self.encryption.kdf_iterations < 100_000:
            raise ConfigurationError(
                "kdf_iterations must be at least 100,000 for security",
                context={
                    "value": self.encryption.kdf_iterations,
                    "minimum": 100_000
                }
            )
        
        # Validate preprocessing config
        if self.preprocessing.target_size[0] <= 0 or self.preprocessing.target_size[1] <= 0:
            raise ConfigurationError(
                "target_size dimensions must be positive",
                context={"value": self.preprocessing.target_size}
            )
        
        if not (0.0 <= self.preprocessing.denoise_strength <= 1.0):
            raise ConfigurationError(
                "denoise_strength must be between 0.0 and 1.0",
                context={
                    "value": self.preprocessing.denoise_strength,
                    "valid_range": "[0.0, 1.0]"
                }
            )
        
        # Validate AI model paths if AI features are enabled
        if self.encryption.use_ai_edge_detection:
            if 'edge_detector' not in self.ai_models:
                raise ConfigurationError(
                    "AI edge detection enabled but 'edge_detector' model path not specified",
                    context={"ai_models": list(self.ai_models.keys())}
                )
            if not self.ai_models['edge_detector'].exists():
                raise ConfigurationError(
                    "Edge detector model file not found",
                    context={"path": str(self.ai_models['edge_detector'])}
                )
        
        if self.encryption.use_ai_optimization:
            if 'coefficient_optimizer' not in self.ai_models:
                raise ConfigurationError(
                    "AI optimization enabled but 'coefficient_optimizer' model path not specified",
                    context={"ai_models": list(self.ai_models.keys())}
                )
            if not self.ai_models['coefficient_optimizer'].exists():
                raise ConfigurationError(
                    "Coefficient optimizer model file not found",
                    context={"path": str(self.ai_models['coefficient_optimizer'])}
                )
        
        if self.encryption.use_anomaly_detection:
            if 'anomaly_detector' not in self.ai_models:
                raise ConfigurationError(
                    "Anomaly detection enabled but 'anomaly_detector' model path not specified",
                    context={"ai_models": list(self.ai_models.keys())}
                )
            if not self.ai_models['anomaly_detector'].exists():
                raise ConfigurationError(
                    "Anomaly detector model file not found",
                    context={"path": str(self.ai_models['anomaly_detector'])}
                )
