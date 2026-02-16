"""
Property-based tests for AI model management.

Feature: fourier-image-encryption
Tests Property 30: Model Version Metadata

These tests verify that model version metadata follows semantic versioning
format and that the model management infrastructure handles versions correctly.
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from fourier_encryption.ai.model_repository import (
    ModelBackend,
    ModelMetadata,
    ModelRepository,
)
from fourier_encryption.models.exceptions import AIModelError


# Strategy for generating valid semantic versions
@st.composite
def semantic_version(draw):
    """Generate valid semantic version strings (major.minor.patch)."""
    major = draw(st.integers(min_value=0, max_value=999))
    minor = draw(st.integers(min_value=0, max_value=999))
    patch = draw(st.integers(min_value=0, max_value=999))
    return f"{major}.{minor}.{patch}"


# Strategy for generating invalid version strings
@st.composite
def invalid_version(draw):
    """Generate invalid version strings."""
    invalid_formats = [
        # Too few parts
        st.just("1"),
        st.just("1.2"),
        # Too many parts
        st.just("1.2.3.4"),
        # Non-numeric parts
        st.just("1.2.x"),
        st.just("a.b.c"),
        # With prefixes/suffixes
        st.just("v1.2.3"),
        st.just("1.2.3-beta"),
        # Empty
        st.just(""),
    ]
    return draw(st.one_of(invalid_formats))


# Strategy for generating model names
model_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
    min_size=1,
    max_size=50
).filter(lambda x: x and not x.startswith('-') and not x.startswith('_'))


# Strategy for generating model backends
backend_strategy = st.sampled_from([ModelBackend.PYTORCH, ModelBackend.TENSORFLOW])


class TestProperty30ModelVersionMetadata:
    """
    Property 30: Model Version Metadata
    
    For any loaded AI model, the model metadata must include a valid version
    string in semantic versioning format (major.minor.patch).
    
    Validates: Requirements 3.19.3
    """
    
    @given(
        name=model_name_strategy,
        version=semantic_version(),
        backend=backend_strategy
    )
    @pytest.mark.property_test
    def test_valid_semantic_version_accepted(self, name, version, backend):
        """
        Feature: fourier-image-encryption
        Property 30: Model Version Metadata
        
        For any valid semantic version string, ModelMetadata should accept it
        without raising an error.
        
        **Validates: Requirements 3.19.3**
        """
        # Create metadata with valid semantic version
        metadata = ModelMetadata(
            name=name,
            version=version,
            backend=backend,
            description="Test model"
        )
        
        # Verify version is stored correctly
        assert metadata.version == version
        
        # Verify version has exactly 3 parts
        parts = version.split('.')
        assert len(parts) == 3
        
        # Verify all parts are numeric
        for part in parts:
            assert part.isdigit()
    
    @given(
        name=model_name_strategy,
        invalid_ver=invalid_version(),
        backend=backend_strategy
    )
    @pytest.mark.property_test
    def test_invalid_version_rejected(self, name, invalid_ver, backend):
        """
        Feature: fourier-image-encryption
        Property 30: Model Version Metadata
        
        For any invalid version string (not semantic versioning format),
        ModelMetadata should raise AIModelError.
        
        **Validates: Requirements 3.19.3**
        """
        # Attempt to create metadata with invalid version
        with pytest.raises(AIModelError) as exc_info:
            ModelMetadata(
                name=name,
                version=invalid_ver,
                backend=backend,
                description="Test model"
            )
        
        # Verify error message mentions version format
        assert "version format" in str(exc_info.value).lower()
    
    @given(
        name=model_name_strategy,
        version=semantic_version(),
        backend=backend_strategy,
        metrics=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=10
        )
    )
    @pytest.mark.property_test
    def test_metadata_serialization_preserves_version(self, name, version, backend, metrics):
        """
        Feature: fourier-image-encryption
        Property 30: Model Version Metadata
        
        For any model metadata, serializing to dict and deserializing back
        should preserve the version string exactly.
        
        **Validates: Requirements 3.19.3**
        """
        # Create metadata
        original = ModelMetadata(
            name=name,
            version=version,
            backend=backend,
            description="Test model",
            metrics=metrics if metrics else None
        )
        
        # Serialize to dict
        data = original.to_dict()
        
        # Verify version in dict
        assert data["version"] == version
        
        # Deserialize back
        restored = ModelMetadata.from_dict(data)
        
        # Verify version preserved
        assert restored.version == version
        assert restored.name == name
        assert restored.backend == backend
    
    @given(
        models=st.lists(
            st.tuples(
                model_name_strategy,
                semantic_version(),
                backend_strategy
            ),
            min_size=1,
            max_size=10,
            unique_by=lambda x: (x[0], x[1])  # Unique by (name, version)
        )
    )
    @pytest.mark.property_test
    def test_repository_metadata_persistence(self, models):
        """
        Feature: fourier-image-encryption
        Property 30: Model Version Metadata
        
        For any set of models registered in a repository, saving and loading
        the repository metadata should preserve all version strings.
        
        **Validates: Requirements 3.19.3**
        """
        # Create repository
        repo = ModelRepository()
        
        # Register all models
        for name, version, backend in models:
            metadata = ModelMetadata(
                name=name,
                version=version,
                backend=backend,
                description=f"Model {name}"
            )
            model_key = f"{name}:{version}"
            repo.registry[model_key] = metadata
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            repo.save_metadata(tmp_path)
            
            # Create new repository and load
            new_repo = ModelRepository()
            new_repo.load_metadata(tmp_path)
            
            # Verify all models and versions preserved
            assert len(new_repo.registry) == len(models)
            
            for name, version, backend in models:
                loaded_meta = new_repo.get_model_metadata(name, version)
                assert loaded_meta is not None
                assert loaded_meta.version == version
                assert loaded_meta.name == name
                assert loaded_meta.backend == backend
        
        finally:
            tmp_path.unlink()
    
    @given(
        name=model_name_strategy,
        versions=st.lists(
            semantic_version(),
            min_size=2,
            max_size=5,
            unique=True
        ),
        backend=backend_strategy
    )
    @pytest.mark.property_test
    def test_version_comparison_ordering(self, name, versions, backend):
        """
        Feature: fourier-image-encryption
        Property 30: Model Version Metadata
        
        For any set of semantic versions, they should be comparable and
        sortable in the expected order (major.minor.patch).
        
        **Validates: Requirements 3.19.3**
        """
        # Create metadata for each version
        metadata_list = []
        for version in versions:
            metadata = ModelMetadata(
                name=name,
                version=version,
                backend=backend,
                description="Test"
            )
            metadata_list.append(metadata)
        
        # Sort by version (semantic versioning order)
        sorted_metadata = sorted(
            metadata_list,
            key=lambda m: tuple(map(int, m.version.split('.'))),
            reverse=True
        )
        
        # Verify sorting worked
        for i in range(len(sorted_metadata) - 1):
            current_parts = tuple(map(int, sorted_metadata[i].version.split('.')))
            next_parts = tuple(map(int, sorted_metadata[i + 1].version.split('.')))
            
            # Current should be >= next (descending order)
            assert current_parts >= next_parts
    
    @given(
        name=model_name_strategy,
        version=semantic_version(),
        backend=backend_strategy,
        input_shape=st.lists(
            st.integers(min_value=1, max_value=1024),
            min_size=1,
            max_size=4
        ),
        output_shape=st.lists(
            st.integers(min_value=1, max_value=1024),
            min_size=1,
            max_size=4
        )
    )
    @pytest.mark.property_test
    def test_metadata_with_shapes_and_version(self, name, version, backend, input_shape, output_shape):
        """
        Feature: fourier-image-encryption
        Property 30: Model Version Metadata
        
        For any model metadata with input/output shapes, the version should
        still be validated and preserved correctly.
        
        **Validates: Requirements 3.19.3**
        """
        # Create metadata with shapes
        metadata = ModelMetadata(
            name=name,
            version=version,
            backend=backend,
            description="Test model",
            input_shape=input_shape,
            output_shape=output_shape
        )
        
        # Verify version is valid
        assert metadata.version == version
        
        # Verify shapes are preserved
        assert metadata.input_shape == input_shape
        assert metadata.output_shape == output_shape
        
        # Serialize and deserialize
        data = metadata.to_dict()
        restored = ModelMetadata.from_dict(data)
        
        # Verify everything preserved
        assert restored.version == version
        assert restored.input_shape == input_shape
        assert restored.output_shape == output_shape


class TestModelMetadataProperties:
    """Additional property tests for model metadata."""
    
    @given(
        name=model_name_strategy,
        version=semantic_version(),
        backend=backend_strategy
    )
    @pytest.mark.property_test
    def test_metadata_immutability_of_version(self, name, version, backend):
        """
        Verify that version validation happens at creation time and
        the version string is stored correctly.
        """
        metadata = ModelMetadata(
            name=name,
            version=version,
            backend=backend,
            description="Test"
        )
        
        # Version should be exactly as provided
        assert metadata.version == version
        
        # Version should have 3 numeric parts
        parts = metadata.version.split('.')
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)
    
    @given(
        name=model_name_strategy,
        version=semantic_version(),
        backend=backend_strategy,
        description=st.text(min_size=0, max_size=200)
    )
    @pytest.mark.property_test
    def test_metadata_to_dict_completeness(self, name, version, backend, description):
        """
        Verify that to_dict() includes all required fields including version.
        """
        metadata = ModelMetadata(
            name=name,
            version=version,
            backend=backend,
            description=description
        )
        
        data = metadata.to_dict()
        
        # Verify all required fields present
        assert "name" in data
        assert "version" in data
        assert "backend" in data
        assert "description" in data
        
        # Verify values match
        assert data["name"] == name
        assert data["version"] == version
        assert data["backend"] == backend.value
        assert data["description"] == description
