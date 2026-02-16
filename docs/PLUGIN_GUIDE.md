# Plugin System

The Fourier-Based Image Encryption System includes a flexible plugin architecture for extensibility.

## Overview

The plugin system enables:
- **Custom Encryption Strategies**: Implement post-quantum algorithms, hardware acceleration, or custom schemes
- **Custom AI Models**: Integrate custom edge detectors, optimizers, and anomaly detectors
- **Future-Ready**: Support for emerging cryptographic standards and AI architectures

## Architecture

### Components

1. **Base Plugin Interfaces** (`base_plugin.py`)
   - `Plugin`: Base interface for all plugins
   - `EncryptionPlugin`: Interface for encryption strategies
   - `AIModelPlugin`: Interface for AI models

2. **Plugin Registries** (`plugin_registry.py`)
   - `EncryptionPluginRegistry`: Manages encryption plugins (singleton)
   - `AIModelPluginRegistry`: Manages AI model plugins (singleton)

3. **Plugin Loader** (`plugin_loader.py`)
   - Automatic plugin discovery from directories
   - Dynamic module loading
   - Plugin validation and registration

### Plugin Lifecycle

```
Discovery → Loading → Registration → Initialization → Usage → Cleanup
```

## Quick Start

### Using Existing Plugins

```python
from fourier_encryption.plugins import PluginLoader

# Create loader
loader = PluginLoader()

# Load plugins from standard locations
loader.load_from_standard_locations(auto_initialize=True)

# Get encryption plugin
encryptor = loader.encryption_registry.get_encryptor("my-plugin")

# Use plugin
encrypted = encryptor.encrypt(data, key)
```

### Creating a Plugin

See `PLUGIN_DEVELOPMENT.md` for detailed guide.

Basic structure:

```python
from fourier_encryption.plugins import EncryptionPlugin, PluginMetadata

class MyEncryptor(EncryptionPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-encryptor",
            version="1.0.0",
            author="Your Name",
            description="My custom encryptor",
            plugin_type="encryption",
            dependencies={}
        )
    
    def initialize(self, config):
        # Setup resources
        pass
    
    def cleanup(self):
        # Release resources
        pass
    
    def derive_key(self, password, salt):
        # Implement key derivation
        pass
    
    def encrypt(self, data, key):
        # Implement encryption
        pass
    
    def decrypt(self, payload, key):
        # Implement decryption
        pass
```

## Plugin Locations

Plugins are automatically discovered from:

- **User plugins**: `~/.fourier_encryption/plugins/`
- **System plugins** (Linux/Mac): `/etc/fourier_encryption/plugins/`
- **System plugins** (Windows): `%APPDATA%/fourier_encryption/plugins/`

## Examples

See `examples/` directory for:
- `example_encryption_plugin.py`: XOR-based encryption (demonstration)
- `example_ai_plugin.py`: Sobel edge detector and simple optimizer

## Extension Points

### 1. Encryption Strategies

Implement custom encryption algorithms:
- Post-quantum cryptography (Kyber, Dilithium, SPHINCS+)
- Hardware-accelerated encryption
- Custom schemes for research

### 2. AI Edge Detection

Integrate custom edge detection models:
- CNN architectures
- Vision Transformers
- Hybrid models
- Domain-specific models

### 3. AI Coefficient Optimization

Implement custom optimization strategies:
- Reinforcement learning
- Neural architecture search
- Custom heuristics

### 4. AI Anomaly Detection

Integrate custom anomaly detectors:
- Deep learning models
- Statistical methods
- Ensemble approaches

## Best Practices

### Security

- Validate all inputs
- Use constant-time operations for crypto
- Never log sensitive data
- Implement proper HMAC verification
- Handle errors securely

### Performance

- Use vectorized operations (NumPy, PyTorch)
- Leverage GPU acceleration when available
- Implement lazy loading
- Clean up resources properly
- Cache frequently used data

### Compatibility

- Specify exact dependency versions
- Support Python 3.9+
- Test on multiple platforms
- Maintain backward compatibility

## Testing

Test your plugins:

```python
import pytest
from fourier_encryption.plugins import PluginLoader

def test_my_plugin():
    loader = PluginLoader()
    loader.load_and_register(Path("path/to/plugin"))
    
    plugin = loader.encryption_registry.get_encryptor("my-plugin")
    assert plugin is not None
    
    # Test encryption/decryption
    data = b"test data"
    key = b"test key" * 4  # 32 bytes
    
    encrypted = plugin.encrypt(data, key)
    decrypted = plugin.decrypt(encrypted, key)
    
    assert decrypted == data
```

## Documentation

- **Plugin Development Guide**: `PLUGIN_DEVELOPMENT.md`
- **API Reference**: See docstrings in source files
- **Examples**: `examples/` directory

## Support

For questions or issues:
- Check `PLUGIN_DEVELOPMENT.md` for detailed documentation
- Review example plugins in `examples/`
- Open an issue on GitHub

## Future Enhancements

Planned features:
- Plugin versioning and compatibility checks
- Plugin dependency resolution
- Hot-reloading of plugins
- Plugin marketplace/registry
- Sandboxed plugin execution
- Plugin performance profiling
