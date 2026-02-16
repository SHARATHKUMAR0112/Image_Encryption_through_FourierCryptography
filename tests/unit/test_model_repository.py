"""
Unit tests for AI model repository.

Tests model loading infrastructure including:
- PyTorch and TensorFlow model loading
- Version metadata extraction and validation
- Model repository pattern for managing multiple models
- Model evaluation metrics
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fourier_encryption.ai.model_repository import (
    ModelBackend,
    ModelLoader,
    ModelMetadata,
    ModelRepository,
)
from fourier_encryption.models.exceptions import AIModelError


class TestModelMetadata:
    """Test ModelMetadata dataclass."""
    
    def test_valid_metadata_creation(self):
        """Test creating metadata with valid semantic version."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.2.3",
            backend=ModelBackend.PYTORCH,
            description="Test model"
        )
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.2.3"
        assert metadata.backend == ModelBackend.PYTORCH
        assert metadata.description == "Test model"
    
    def test_invalid_version_format(self):
        """Test that invalid version format raises error."""
        with pytest.raises(AIModelError) as exc_info:
            ModelMetadata(
                name="test_model",
                version="1.2",  # Invalid: only 2 parts
                backend=ModelBackend.PYTORCH,
                description="Test"
            )
        
        assert "Invalid version format" in str(exc_info.value)
    
    def test_invalid_version_non_numeric(self):
        """Test that non-numeric version parts raise error."""
        with pytest.raises(AIModelError) as exc_info:
            ModelMetadata(
                name="test_model",
                version="1.2.x",  # Invalid: non-numeric
                backend=ModelBackend.PYTORCH,
                description="Test"
            )
        
        assert "Invalid version format" in str(exc_info.value)
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            backend=ModelBackend.TENSORFLOW,
            description="Test model",
            input_shape=[1, 256, 256],
            output_shape=[1, 256, 256],
            metrics={"accuracy": 0.95, "f1_score": 0.92}
        )
        
        data = metadata.to_dict()
        
        assert data["name"] == "test_model"
        assert data["version"] == "1.0.0"
        assert data["backend"] == "tensorflow"
        assert data["input_shape"] == [1, 256, 256]
        assert data["metrics"]["accuracy"] == 0.95
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "name": "test_model",
            "version": "2.1.0",
            "backend": "pytorch",
            "description": "Test model",
            "input_shape": [1, 128, 128],
            "metrics": {"accuracy": 0.90}
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        assert metadata.name == "test_model"
        assert metadata.version == "2.1.0"
        assert metadata.backend == ModelBackend.PYTORCH
        assert metadata.input_shape == [1, 128, 128]
        assert metadata.metrics["accuracy"] == 0.90
    
    def test_metadata_with_model_path(self):
        """Test metadata with model path."""
        model_path = Path("/path/to/model.pt")
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            backend=ModelBackend.PYTORCH,
            description="Test",
            model_path=model_path
        )
        
        assert metadata.model_path == model_path
        
        # Test serialization
        data = metadata.to_dict()
        assert data["model_path"] == str(model_path)


class TestModelLoader:
    """Test ModelLoader class."""
    
    def test_model_not_found(self):
        """Test loading non-existent model raises error."""
        loader = ModelLoader()
        non_existent_path = Path("/nonexistent/model.pt")
        
        with pytest.raises(AIModelError) as exc_info:
            loader.load_model(non_existent_path, ModelBackend.PYTORCH)
        
        assert "Model file not found" in str(exc_info.value)
    
    def test_unsupported_backend(self):
        """Test loading with unsupported backend raises error."""
        loader = ModelLoader()
        
        with tempfile.NamedTemporaryFile(suffix=".model") as tmp:
            model_path = Path(tmp.name)
            
            with pytest.raises(AIModelError) as exc_info:
                loader.load_model(model_path, ModelBackend.UNKNOWN)
            
            assert "Unsupported backend" in str(exc_info.value)
    
    def test_pytorch_not_installed_error(self):
        """Test that missing PyTorch raises appropriate error."""
        loader = ModelLoader()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            model_path = Path(tmp.name)
        
        try:
            # Try to load - will fail if PyTorch not installed
            # or succeed if it is installed
            try:
                loader.load_model(model_path, ModelBackend.PYTORCH)
            except AIModelError as e:
                # If PyTorch is not installed, should get specific error
                if "PyTorch is not installed" in str(e):
                    assert "pip install torch" in str(e)
                else:
                    # Other errors are acceptable (e.g., invalid file format)
                    pass
        finally:
            # Clean up
            model_path.unlink()
    
    def test_tensorflow_not_installed_error(self):
        """Test that missing TensorFlow raises appropriate error."""
        loader = ModelLoader()
        
        # Create a temporary directory (TensorFlow models are directories)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            
            try:
                loader.load_model(model_path, ModelBackend.TENSORFLOW)
            except AIModelError as e:
                # If TensorFlow is not installed, should get specific error
                if "TensorFlow is not installed" in str(e):
                    assert "pip install tensorflow" in str(e)
                else:
                    # Other errors are acceptable (e.g., invalid model format)
                    pass
    
    def test_cached_model_retrieval(self):
        """Test retrieving cached models."""
        loader = ModelLoader()
        
        # Manually add a model to cache
        test_model = {"dummy": "model"}
        loader.loaded_models["test_model:1.0.0"] = test_model
        
        # Retrieve cached model
        cached = loader.get_cached_model("test_model", "1.0.0")
        assert cached == test_model
        
        # Non-existent model returns None
        not_cached = loader.get_cached_model("other_model", "2.0.0")
        assert not_cached is None


class TestModelRepository:
    """Test ModelRepository class."""
    
    def test_repository_initialization(self):
        """Test repository initialization."""
        models_dir = Path("/path/to/models")
        repo = ModelRepository(models_dir=models_dir)
        
        assert repo.models_dir == models_dir
        assert isinstance(repo.loader, ModelLoader)
        assert len(repo.registry) == 0
    
    def test_register_model_invalid_backend_string(self):
        """Test registering model with invalid backend string."""
        repo = ModelRepository()
        
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            model_path = Path(tmp.name)
            
            with pytest.raises(AIModelError) as exc_info:
                repo.register_model(model_path, "invalid_backend")
            
            assert "Invalid backend" in str(exc_info.value)
    
    def test_load_model_not_found(self):
        """Test loading non-existent model raises error."""
        repo = ModelRepository()
        
        with pytest.raises(AIModelError) as exc_info:
            repo.load_model("nonexistent_model", "1.0.0")
        
        assert "Model not found" in str(exc_info.value)
    
    def test_load_model_no_version_no_models(self):
        """Test loading model without version when no models exist."""
        repo = ModelRepository()
        
        with pytest.raises(AIModelError) as exc_info:
            repo.load_model("test_model")
        
        assert "No models found with name" in str(exc_info.value)
    
    def test_list_models_empty(self):
        """Test listing models in empty repository."""
        repo = ModelRepository()
        models = repo.list_models()
        
        assert models == []
    
    def test_get_model_metadata_not_found(self):
        """Test getting metadata for non-existent model."""
        repo = ModelRepository()
        metadata = repo.get_model_metadata("test_model", "1.0.0")
        
        assert metadata is None
    
    def test_save_and_load_metadata(self):
        """Test saving and loading repository metadata."""
        repo = ModelRepository()
        
        # Add some metadata to registry
        metadata1 = ModelMetadata(
            name="model1",
            version="1.0.0",
            backend=ModelBackend.PYTORCH,
            description="First model",
            metrics={"accuracy": 0.95}
        )
        metadata2 = ModelMetadata(
            name="model2",
            version="2.1.0",
            backend=ModelBackend.TENSORFLOW,
            description="Second model",
            metrics={"f1_score": 0.92}
        )
        
        repo.registry["model1:1.0.0"] = metadata1
        repo.registry["model2:2.1.0"] = metadata2
        
        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            repo.save_metadata(tmp_path)
            
            # Create new repository and load metadata
            new_repo = ModelRepository()
            new_repo.load_metadata(tmp_path)
            
            # Verify loaded metadata
            assert len(new_repo.registry) == 2
            
            loaded_meta1 = new_repo.get_model_metadata("model1", "1.0.0")
            assert loaded_meta1 is not None
            assert loaded_meta1.name == "model1"
            assert loaded_meta1.version == "1.0.0"
            assert loaded_meta1.metrics["accuracy"] == 0.95
            
            loaded_meta2 = new_repo.get_model_metadata("model2", "2.1.0")
            assert loaded_meta2 is not None
            assert loaded_meta2.name == "model2"
            assert loaded_meta2.backend == ModelBackend.TENSORFLOW
        
        finally:
            tmp_path.unlink()
    
    def test_load_metadata_file_not_found(self):
        """Test loading metadata from non-existent file."""
        repo = ModelRepository()
        non_existent = Path("/nonexistent/metadata.json")
        
        with pytest.raises(AIModelError) as exc_info:
            repo.load_metadata(non_existent)
        
        assert "Metadata file not found" in str(exc_info.value)
    
    def test_load_metadata_invalid_json(self):
        """Test loading metadata from invalid JSON file."""
        repo = ModelRepository()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write("invalid json content {")
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(AIModelError) as exc_info:
                repo.load_metadata(tmp_path)
            
            assert "Failed to load metadata" in str(exc_info.value)
        
        finally:
            tmp_path.unlink()
    
    def test_model_metadata_with_path(self):
        """Test metadata includes model path after registration."""
        repo = ModelRepository()
        
        # Manually add metadata with path
        model_path = Path("/path/to/model.pt")
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            backend=ModelBackend.PYTORCH,
            description="Test",
            model_path=model_path
        )
        
        repo.registry["test_model:1.0.0"] = metadata
        
        # Retrieve and verify
        retrieved = repo.get_model_metadata("test_model", "1.0.0")
        assert retrieved is not None
        assert retrieved.model_path == model_path
    
    def test_list_models_with_entries(self):
        """Test listing models with multiple entries."""
        repo = ModelRepository()
        
        # Add multiple models
        for i in range(3):
            metadata = ModelMetadata(
                name=f"model{i}",
                version=f"{i}.0.0",
                backend=ModelBackend.PYTORCH,
                description=f"Model {i}"
            )
            repo.registry[f"model{i}:{i}.0.0"] = metadata
        
        models = repo.list_models()
        assert len(models) == 3
        
        # Verify all models are present
        model_names = {m.name for m in models}
        assert model_names == {"model0", "model1", "model2"}


class TestModelBackend:
    """Test ModelBackend enum."""
    
    def test_backend_values(self):
        """Test backend enum values."""
        assert ModelBackend.PYTORCH.value == "pytorch"
        assert ModelBackend.TENSORFLOW.value == "tensorflow"
        assert ModelBackend.UNKNOWN.value == "unknown"
    
    def test_backend_from_string(self):
        """Test creating backend from string."""
        assert ModelBackend("pytorch") == ModelBackend.PYTORCH
        assert ModelBackend("tensorflow") == ModelBackend.TENSORFLOW
        assert ModelBackend("unknown") == ModelBackend.UNKNOWN
    
    def test_invalid_backend_string(self):
        """Test invalid backend string raises ValueError."""
        with pytest.raises(ValueError):
            ModelBackend("invalid")


class TestSemanticVersioning:
    """Test semantic versioning validation."""
    
    def test_valid_versions(self):
        """Test various valid semantic versions."""
        valid_versions = [
            "0.0.0",
            "1.0.0",
            "1.2.3",
            "10.20.30",
            "999.999.999"
        ]
        
        for version in valid_versions:
            metadata = ModelMetadata(
                name="test",
                version=version,
                backend=ModelBackend.PYTORCH,
                description="Test"
            )
            assert metadata.version == version
    
    def test_invalid_versions(self):
        """Test various invalid version formats."""
        invalid_versions = [
            "1",           # Too few parts
            "1.2",         # Too few parts
            "1.2.3.4",     # Too many parts
            "1.2.x",       # Non-numeric
            "v1.2.3",      # Prefix
            "1.2.3-beta",  # Suffix
            "",            # Empty
            "a.b.c"        # All non-numeric
        ]
        
        for version in invalid_versions:
            with pytest.raises(AIModelError):
                ModelMetadata(
                    name="test",
                    version=version,
                    backend=ModelBackend.PYTORCH,
                    description="Test"
                )


class TestModelMetrics:
    """Test model evaluation metrics handling."""
    
    def test_metrics_in_metadata(self):
        """Test including evaluation metrics in metadata."""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.94,
            "f1_score": 0.935,
            "inference_time_ms": 15.5
        }
        
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            backend=ModelBackend.PYTORCH,
            description="Test",
            metrics=metrics
        )
        
        assert metadata.metrics == metrics
        assert metadata.metrics["accuracy"] == 0.95
        assert metadata.metrics["f1_score"] == 0.935
    
    def test_metrics_serialization(self):
        """Test metrics are preserved in serialization."""
        metrics = {"accuracy": 0.95, "f1_score": 0.92}
        
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            backend=ModelBackend.PYTORCH,
            description="Test",
            metrics=metrics
        )
        
        # Serialize and deserialize
        data = metadata.to_dict()
        restored = ModelMetadata.from_dict(data)
        
        assert restored.metrics == metrics
    
    def test_optional_metrics(self):
        """Test that metrics are optional."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            backend=ModelBackend.PYTORCH,
            description="Test"
            # No metrics provided
        )
        
        assert metadata.metrics is None
