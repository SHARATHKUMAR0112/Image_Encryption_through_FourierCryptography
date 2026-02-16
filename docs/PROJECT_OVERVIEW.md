# Fourier-Based Image Encryption System with AI Integration

An industrial-grade image encryption system that uses Fourier Series-based sketch reconstruction with epicycles as the core encryption mechanism, enhanced with AI capabilities for intelligent optimization and security.

## Features

- **Fourier Transform Encryption**: Uses Discrete Fourier Transform to decompose images into frequency components
- **Epicycle Animation**: Visual reconstruction through rotating epicycles
- **AES-256 Encryption**: Military-grade encryption for Fourier coefficients
- **AI-Enhanced Processing**: Intelligent edge detection, coefficient optimization, and anomaly detection
- **Real-time Visualization**: Live animation of the encryption/decryption process
- **Monitoring Dashboard**: Real-time metrics and performance tracking
- **Clean Architecture**: Modular design following SOLID principles

## Project Structure

```
fourier_encryption/
├── core/                  # Core image processing and Fourier transform
├── encryption/            # Encryption and key management
├── visualization/         # Live rendering and monitoring
├── transmission/          # Serialization and secure channels
├── ai/                    # AI-enhanced components
├── models/                # Data models and structures
├── config/                # Configuration management
│   ├── logging_config.py  # Automatic timestamped logging
│   └── paths.py           # Path configuration for images and logs
├── cli/                   # Command-line interface
├── api/                   # REST API (optional)
tests/
├── unit/                  # Unit tests
├── integration/           # Integration tests
├── fixtures/              # Test data and fixtures (includes image fixtures)
images/                    # Test images for encryption/decryption
logs/                      # Timestamped log files (auto-generated)
examples/                  # Example scripts and demonstrations
```

## Test Images

The project includes test images in the `images/` folder for demonstrations and testing:

- **luffy_1.png**, **luffy2.png** - Anime character images
- **img1_prettylady.jpg**, **lady2.jpeg**, **lady3.jpg** - Portrait photos
- **img_2_malebpy.webp** - Male portrait photo

### Using Test Images

```python
from fourier_encryption.config.paths import get_test_image, list_test_images

# Get a specific image
image_path = get_test_image("luffy_1.png")

# List all available images
all_images = list_test_images()
```

See `images/README.md` for more details.

## Automatic Logging

All terminal output is automatically logged to timestamped files in the `logs/` folder:

- **fourier_encryption_YYYYMMDD_HHMMSS.log** - Timestamped log for each run
- **latest.log** - Always contains the most recent run

Sensitive information (keys, passwords) is automatically redacted from logs.

See `logs/README.md` for more details.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Basic Installation

```bash
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### With AI Components

```bash
pip install -e ".[ai]"
```

### With API Support

```bash
pip install -e ".[api]"
```

### Full Installation

```bash
pip install -e ".[all]"
```

## Quick Start

### Basic Encryption/Decryption

```python
from fourier_encryption.config.logging_config import setup_logging

# Initialize logging
setup_logging(level="INFO")

# More functionality will be added in subsequent tasks
```

### Concurrent Encryption

Encrypt multiple images in parallel using the concurrent orchestrator:

```python
from fourier_encryption.concurrency import (
    ConcurrentEncryptionOrchestrator,
    EncryptionTask,
)
from fourier_encryption.config.settings import EncryptionConfig, PreprocessConfig

# Create concurrent orchestrator with 4 worker threads
concurrent_orchestrator = ConcurrentEncryptionOrchestrator(
    orchestrator=base_orchestrator,
    max_workers=4,
    progress_callback=lambda task_id, progress: print(f"{task_id}: {progress}%"),
)

# Prepare encryption tasks
tasks = [
    EncryptionTask(
        image_path=Path(f"image_{i}.png"),
        key=f"password_{i}",
        preprocess_config=PreprocessConfig(),
        encryption_config=EncryptionConfig(num_coefficients=100),
        task_id=f"task_{i}",
    )
    for i in range(10)
]

# Execute batch encryption (blocks until all complete)
results = concurrent_orchestrator.encrypt_batch(tasks, wait=True)

# Check results
for result in results:
    if result.success:
        print(f"{result.task_id}: Success - {len(result.payload.ciphertext)} bytes")
    else:
        print(f"{result.task_id}: Failed - {result.error}")
```

See `examples/concurrent_encryption_demo.py` for a complete example.

## Configuration

The system uses YAML or JSON configuration files. Example configuration:

```yaml
encryption:
  num_coefficients: null  # Auto-optimize
  use_ai_edge_detection: true
  use_ai_optimization: true
  kdf_iterations: 100000

preprocessing:
  target_size: [1920, 1080]
  maintain_aspect_ratio: true
  normalize: true

logging:
  level: INFO
  file: logs/fourier_encryption.log
```

## Development

### Running Tests

```bash
pytest
```

### Code Coverage

```bash
pytest --cov=fourier_encryption --cov-report=html
```

### Code Formatting

```bash
black fourier_encryption tests
```

### Type Checking

```bash
mypy fourier_encryption
```

## Architecture

The system follows Clean Architecture principles with clear layer separation:

1. **Presentation Layer**: CLI, API, Visualization
2. **Application Layer**: Orchestration and workflows
3. **Domain Layer**: Business logic, encryption, AI
4. **Infrastructure Layer**: File I/O, serialization, crypto libraries

## Plugin System

The system includes a flexible plugin architecture for extensibility:

### Features

- **Custom Encryption Strategies**: Implement post-quantum algorithms (Kyber, Dilithium), hardware acceleration, or custom schemes
- **Custom AI Models**: Integrate custom edge detectors, optimizers, and anomaly detectors
- **Automatic Discovery**: Plugins are automatically discovered from standard locations
- **Hot-Loading**: Load plugins dynamically without restarting the system

### Plugin Locations

Plugins are automatically discovered from:
- User plugins: `~/.fourier_encryption/plugins/`
- System plugins (Linux/Mac): `/etc/fourier_encryption/plugins/`
- System plugins (Windows): `%APPDATA%/fourier_encryption/plugins/`

### Quick Start

```python
from fourier_encryption.plugins import PluginLoader

# Load plugins from standard locations
loader = PluginLoader()
loader.load_from_standard_locations(auto_initialize=True)

# Get encryption plugin
encryptor = loader.encryption_registry.get_encryptor("my-plugin")

# Use plugin
encrypted = encryptor.encrypt(data, key)
```

### Creating a Plugin

See `fourier_encryption/plugins/PLUGIN_DEVELOPMENT.md` for detailed guide.

Basic encryption plugin structure:

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
    
    def encrypt(self, data, key):
        # Implement encryption
        pass
    
    def decrypt(self, payload, key):
        # Implement decryption
        pass
```

### Example Plugins

The system includes example plugins in `fourier_encryption/plugins/examples/`:
- `example_encryption_plugin.py`: XOR-based encryption (demonstration)
- `example_ai_plugin.py`: Sobel edge detector and simple optimizer

Run the demo:
```bash
python examples/plugin_system_demo.py
```

### Extension Points

1. **Encryption Strategies**: Post-quantum crypto, hardware acceleration
2. **AI Edge Detection**: Custom CNN, Vision Transformers, domain-specific models
3. **AI Coefficient Optimization**: RL agents, neural architecture search
4. **AI Anomaly Detection**: Deep learning, statistical models, ensemble methods

See `fourier_encryption/plugins/README.md` for more details.

## Security

- AES-256-GCM encryption
- PBKDF2-HMAC-SHA256 key derivation (100,000+ iterations)
- Cryptographically secure random number generation
- Constant-time operations for key comparison
- Automatic sensitive data sanitization in logs

## License

MIT License

## Contributing

Contributions are welcome! Please ensure all tests pass and code follows PEP8 guidelines.

## Status

🚧 **Under Development** - Core infrastructure complete, implementing features incrementally.
