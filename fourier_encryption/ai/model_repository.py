"""
AI Model Repository for managing multiple AI models.

This module implements the Repository pattern for loading, managing, and
versioning AI models used in the Fourier-Based Image Encryption System.
Supports both PyTorch and TensorFlow backends.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fourier_encryption.models.exceptions import AIModelError

logger = logging.getLogger(__name__)


class ModelBackend(Enum):
    """Supported AI model backends."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    UNKNOWN = "unknown"


@dataclass
class ModelMetadata:
    """
    Metadata for an AI model.
    
    Attributes:
        name: Model name/identifier
        version: Semantic version string (e.g., "1.2.3")
        backend: Model framework (PyTorch or TensorFlow)
        description: Human-readable description
        input_shape: Expected input shape (e.g., [1, 256, 256])
        output_shape: Expected output shape
        metrics: Performance metrics (accuracy, F1-score, etc.)
        created_at: Creation timestamp
        model_path: Path to model file
    """
    name: str
    version: str
    backend: ModelBackend
    description: str
    input_shape: Optional[list] = None
    output_shape: Optional[list] = None
    metrics: Optional[Dict[str, float]] = None
    created_at: Optional[str] = None
    model_path: Optional[Path] = None
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        # Validate version format (semantic versioning)
        if not self._is_valid_semver(self.version):
            raise AIModelError(
                f"Invalid version format: {self.version}. "
                "Expected semantic versioning (e.g., '1.2.3')"
            )
    
    @staticmethod
    def _is_valid_semver(version: str) -> bool:
        """
        Validate semantic versioning format.
        
        Args:
            version: Version string to validate
        
        Returns:
            True if valid semantic version, False otherwise
        """
        parts = version.split('.')
        if len(parts) != 3:
            return False
        
        try:
            # All parts must be non-negative integers
            major, minor, patch = parts
            int(major), int(minor), int(patch)
            return True
        except ValueError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "backend": self.backend.value,
            "description": self.description,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "model_path": str(self.model_path) if self.model_path else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        backend_str = data.get("backend", "unknown")
        try:
            backend = ModelBackend(backend_str)
        except ValueError:
            backend = ModelBackend.UNKNOWN
        
        model_path = data.get("model_path")
        if model_path:
            model_path = Path(model_path)
        
        return cls(
            name=data["name"],
            version=data["version"],
            backend=backend,
            description=data.get("description", ""),
            input_shape=data.get("input_shape"),
            output_shape=data.get("output_shape"),
            metrics=data.get("metrics"),
            created_at=data.get("created_at"),
            model_path=model_path
        )


class ModelLoader:
    """
    Base class for model loading.
    
    Provides common functionality for loading AI models from disk
    with version validation and metadata extraction.
    """
    
    def __init__(self):
        """Initialize model loader."""
        self.loaded_models: Dict[str, Any] = {}
    
    def load_model(
        self,
        model_path: Path,
        backend: ModelBackend
    ) -> tuple[Any, ModelMetadata]:
        """
        Load model from disk with metadata extraction.
        
        Args:
            model_path: Path to model file
            backend: Model framework (PyTorch or TensorFlow)
        
        Returns:
            Tuple of (loaded_model, metadata)
        
        Raises:
            AIModelError: If model loading fails
        """
        if not model_path.exists():
            raise AIModelError(
                f"Model file not found: {model_path}",
                context={"path": str(model_path)}
            )
        
        logger.info(f"Loading model from {model_path} (backend: {backend.value})")
        
        try:
            if backend == ModelBackend.PYTORCH:
                model, metadata = self._load_pytorch_model(model_path)
            elif backend == ModelBackend.TENSORFLOW:
                model, metadata = self._load_tensorflow_model(model_path)
            else:
                raise AIModelError(
                    f"Unsupported backend: {backend}",
                    context={"backend": backend.value}
                )
            
            # Cache loaded model
            model_key = f"{metadata.name}:{metadata.version}"
            self.loaded_models[model_key] = model
            
            logger.info(
                f"Successfully loaded model: {metadata.name} v{metadata.version}"
            )
            
            return model, metadata
            
        except Exception as e:
            if isinstance(e, AIModelError):
                raise
            raise AIModelError(
                f"Failed to load model from {model_path}: {e}",
                context={"path": str(model_path), "error": str(e)}
            ) from e
    
    def _load_pytorch_model(self, model_path: Path) -> tuple[Any, ModelMetadata]:
        """
        Load PyTorch model from disk.
        
        Args:
            model_path: Path to PyTorch model file (.pt or .pth)
        
        Returns:
            Tuple of (model, metadata)
        
        Raises:
            AIModelError: If PyTorch is not available or loading fails
        """
        try:
            import torch
        except ImportError:
            raise AIModelError(
                "PyTorch is not installed. Install with: pip install torch",
                context={"backend": "pytorch"}
            )
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract metadata if available
            if isinstance(checkpoint, dict) and 'metadata' in checkpoint:
                metadata_dict = checkpoint['metadata']
                metadata = ModelMetadata.from_dict(metadata_dict)
                metadata.model_path = model_path
                metadata.backend = ModelBackend.PYTORCH
                
                # Extract model from checkpoint
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    # State dict only - need model architecture
                    logger.warning(
                        "Checkpoint contains only state_dict, "
                        "model architecture not included"
                    )
                    model = checkpoint['model_state_dict']
                else:
                    model = checkpoint
            else:
                # No metadata, create default
                model = checkpoint
                metadata = ModelMetadata(
                    name=model_path.stem,
                    version="0.0.0",
                    backend=ModelBackend.PYTORCH,
                    description="PyTorch model (no metadata)",
                    model_path=model_path
                )
            
            return model, metadata
            
        except Exception as e:
            raise AIModelError(
                f"Failed to load PyTorch model: {e}",
                context={"path": str(model_path), "error": str(e)}
            ) from e
    
    def _load_tensorflow_model(
        self,
        model_path: Path
    ) -> tuple[Any, ModelMetadata]:
        """
        Load TensorFlow model from disk.
        
        Args:
            model_path: Path to TensorFlow model directory or .h5 file
        
        Returns:
            Tuple of (model, metadata)
        
        Raises:
            AIModelError: If TensorFlow is not available or loading fails
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise AIModelError(
                "TensorFlow is not installed. Install with: pip install tensorflow",
                context={"backend": "tensorflow"}
            )
        
        try:
            # Load model
            model = tf.keras.models.load_model(str(model_path))
            
            # Try to load metadata from companion JSON file
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)
                metadata.model_path = model_path
                metadata.backend = ModelBackend.TENSORFLOW
            else:
                # Create default metadata
                metadata = ModelMetadata(
                    name=model_path.stem,
                    version="0.0.0",
                    backend=ModelBackend.TENSORFLOW,
                    description="TensorFlow model (no metadata)",
                    input_shape=list(model.input_shape) if hasattr(model, 'input_shape') else None,
                    output_shape=list(model.output_shape) if hasattr(model, 'output_shape') else None,
                    model_path=model_path
                )
            
            return model, metadata
            
        except Exception as e:
            raise AIModelError(
                f"Failed to load TensorFlow model: {e}",
                context={"path": str(model_path), "error": str(e)}
            ) from e
    
    def get_cached_model(self, name: str, version: str) -> Optional[Any]:
        """
        Get cached model if already loaded.
        
        Args:
            name: Model name
            version: Model version
        
        Returns:
            Cached model or None if not found
        """
        model_key = f"{name}:{version}"
        return self.loaded_models.get(model_key)


class ModelRepository:
    """
    Repository for managing multiple AI models.
    
    Provides centralized model management with support for:
    - Loading models from disk (PyTorch and TensorFlow)
    - Version metadata extraction and validation
    - Model caching to avoid redundant loading
    - Model evaluation metrics tracking
    
    Attributes:
        models_dir: Base directory for model files
        loader: ModelLoader instance for loading operations
        registry: Dictionary mapping model names to metadata
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize model repository.
        
        Args:
            models_dir: Base directory for model files (optional)
        """
        self.models_dir = models_dir
        self.loader = ModelLoader()
        self.registry: Dict[str, ModelMetadata] = {}
        
        logger.info(f"Initialized ModelRepository with models_dir: {models_dir}")
    
    def register_model(
        self,
        model_path: Path,
        backend: Union[ModelBackend, str],
        metadata: Optional[ModelMetadata] = None
    ) -> ModelMetadata:
        """
        Register a model in the repository.
        
        Args:
            model_path: Path to model file
            backend: Model framework (PyTorch or TensorFlow)
            metadata: Optional pre-existing metadata
        
        Returns:
            ModelMetadata for the registered model
        
        Raises:
            AIModelError: If registration fails
        """
        # Convert string backend to enum
        if isinstance(backend, str):
            try:
                backend = ModelBackend(backend.lower())
            except ValueError:
                raise AIModelError(
                    f"Invalid backend: {backend}",
                    context={"valid_backends": [b.value for b in ModelBackend]}
                )
        
        # Load model to extract/validate metadata
        model, extracted_metadata = self.loader.load_model(model_path, backend)
        
        # Use provided metadata if available, otherwise use extracted
        if metadata is not None:
            # Validate provided metadata matches extracted
            if metadata.version != extracted_metadata.version:
                logger.warning(
                    f"Provided version {metadata.version} differs from "
                    f"extracted version {extracted_metadata.version}"
                )
            final_metadata = metadata
        else:
            final_metadata = extracted_metadata
        
        # Register in repository
        model_key = f"{final_metadata.name}:{final_metadata.version}"
        self.registry[model_key] = final_metadata
        
        logger.info(f"Registered model: {model_key}")
        
        return final_metadata
    
    def load_model(
        self,
        name: str,
        version: Optional[str] = None
    ) -> tuple[Any, ModelMetadata]:
        """
        Load a model from the repository.
        
        Args:
            name: Model name
            version: Model version (optional, uses latest if not specified)
        
        Returns:
            Tuple of (model, metadata)
        
        Raises:
            AIModelError: If model not found or loading fails
        """
        # Find matching model in registry
        if version:
            model_key = f"{name}:{version}"
            if model_key not in self.registry:
                raise AIModelError(
                    f"Model not found: {name} v{version}",
                    context={"name": name, "version": version}
                )
            metadata = self.registry[model_key]
        else:
            # Find latest version
            matching_models = [
                (key, meta) for key, meta in self.registry.items()
                if meta.name == name
            ]
            if not matching_models:
                raise AIModelError(
                    f"No models found with name: {name}",
                    context={"name": name}
                )
            
            # Sort by version and get latest
            matching_models.sort(
                key=lambda x: tuple(map(int, x[1].version.split('.'))),
                reverse=True
            )
            metadata = matching_models[0][1]
        
        # Check if already cached
        cached_model = self.loader.get_cached_model(metadata.name, metadata.version)
        if cached_model is not None:
            logger.debug(f"Using cached model: {metadata.name} v{metadata.version}")
            return cached_model, metadata
        
        # Load from disk
        if metadata.model_path is None:
            raise AIModelError(
                f"Model path not set for {metadata.name} v{metadata.version}",
                context={"name": metadata.name, "version": metadata.version}
            )
        
        model, _ = self.loader.load_model(metadata.model_path, metadata.backend)
        return model, metadata
    
    def list_models(self) -> list[ModelMetadata]:
        """
        List all registered models.
        
        Returns:
            List of ModelMetadata for all registered models
        """
        return list(self.registry.values())
    
    def get_model_metadata(self, name: str, version: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a specific model.
        
        Args:
            name: Model name
            version: Model version
        
        Returns:
            ModelMetadata or None if not found
        """
        model_key = f"{name}:{version}"
        return self.registry.get(model_key)
    
    def save_metadata(self, output_path: Path) -> None:
        """
        Save repository metadata to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        metadata_list = [meta.to_dict() for meta in self.registry.values()]
        
        with open(output_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(f"Saved repository metadata to {output_path}")
    
    def load_metadata(self, input_path: Path) -> None:
        """
        Load repository metadata from JSON file.
        
        Args:
            input_path: Path to input JSON file
        
        Raises:
            AIModelError: If loading fails
        """
        if not input_path.exists():
            raise AIModelError(
                f"Metadata file not found: {input_path}",
                context={"path": str(input_path)}
            )
        
        try:
            with open(input_path, 'r') as f:
                metadata_list = json.load(f)
            
            for metadata_dict in metadata_list:
                metadata = ModelMetadata.from_dict(metadata_dict)
                model_key = f"{metadata.name}:{metadata.version}"
                self.registry[model_key] = metadata
            
            logger.info(
                f"Loaded {len(metadata_list)} model metadata entries from {input_path}"
            )
            
        except Exception as e:
            raise AIModelError(
                f"Failed to load metadata from {input_path}: {e}",
                context={"path": str(input_path), "error": str(e)}
            ) from e
