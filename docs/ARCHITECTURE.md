# System Architecture

## Overview

The Fourier-Based Image Encryption System follows Clean Architecture principles with strict layer separation, SOLID design principles, and a plugin-based extensibility model.

## Core Design Principles

1. **Clean Architecture**: Strict layer separation (Presentation → Application → Domain → Infrastructure)
2. **SOLID Principles**: Single responsibility, dependency inversion, interface segregation
3. **Security by Design**: Defense in depth, fail-secure defaults, constant-time operations
4. **Performance First**: Vectorized operations, GPU acceleration, parallel processing
5. **Extensibility**: Plugin architecture for algorithms, AI models, and encryption strategies

## Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CLI Module  │  │  REST API    │  │ Visualization │      │
│  │   (Typer)    │  │  (FastAPI)   │  │  Dashboard    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Encryption Orchestrator                      │   │
│  │  - Coordinates encryption/decryption workflows       │   │
│  │  - Manages AI model lifecycle                        │   │
│  │  - Handles monitoring and metrics                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      DOMAIN LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Image Pipeline│  │Fourier Engine│  │Encryption    │      │
│  │- Preprocessing│  │- DFT/IDFT    │  │- AES-256     │      │
│  │- Edge Detect │  │- Epicycles   │  │- Key Derive  │      │
│  │- Contours    │  │- Reconstruct │  │- HMAC        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │AI Edge Model │  │AI Optimizer  │  │AI Anomaly    │      │
│  │- CNN/ViT     │  │- Coefficient │  │- Tamper      │      │
│  │- GPU Accel   │  │- Selection   │  │- Detection   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Serialization │  │Crypto Library│  │File I/O      │      │
│  │- MessagePack │  │- cryptography│  │- Image Load  │      │
│  │- Validation  │  │- Secure RNG  │  │- Config      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Design Patterns

### 1. Strategy Pattern
Used for encryption algorithms and edge detection strategies.

```python
class EncryptionStrategy(ABC):
    @abstractmethod
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        pass
    
    @abstractmethod
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        pass

# Implementations: AES256Encryptor, future post-quantum algorithms
```

### 2. Factory Pattern
Creates appropriate encryptors and processors based on configuration.

```python
class EncryptorFactory:
    @staticmethod
    def create(algorithm: str) -> EncryptionStrategy:
        if algorithm == "aes256":
            return AES256Encryptor()
        # Future: Kyber, Dilithium, etc.
```

### 3. Observer Pattern
Live visualization updates during epicycle animation.

```python
class LiveRenderer:
    def attach_observer(self, observer: RenderObserver):
        self.observers.append(observer)
    
    def notify_observers(self, state: EpicycleState):
        for observer in self.observers:
            observer.update(state)
```

### 4. Repository Pattern
AI model loading and management.

```python
class ModelRepository:
    def load_model(self, name: str, version: str) -> torch.nn.Module:
        # Load from disk with version validation
        pass
```

### 5. Dependency Injection
All components receive dependencies via constructors.

```python
class EncryptionOrchestrator:
    def __init__(self,
                 image_processor: ImageProcessor,
                 edge_detector: EdgeDetector,
                 fourier_transformer: FourierTransformer,
                 encryptor: EncryptionStrategy,
                 serializer: CoefficientSerializer):
        # Dependencies injected, not created internally
        self.image_processor = image_processor
        self.edge_detector = edge_detector
        # ...
```

## Data Flow

### Encryption Pipeline

```
Image File
    ↓
[ImageProcessor] Load & Preprocess
    ↓
[EdgeDetector] Extract Edges (Canny/Industrial/AI)
    ↓
[ContourExtractor] Find Contours
    ↓
[FourierTransformer] Compute DFT
    ↓
[CoefficientOptimizer] AI Optimization (optional)
    ↓
[FourierTransformer] Sort by Amplitude
    ↓
[CoefficientSerializer] Serialize to MessagePack
    ↓
[AES256Encryptor] Encrypt + HMAC
    ↓
Encrypted Payload (JSON)
```

### Decryption Pipeline

```
Encrypted Payload (JSON)
    ↓
[AnomalyDetector] Check for Tampering (optional)
    ↓
[AES256Encryptor] Verify HMAC & Decrypt
    ↓
[CoefficientSerializer] Deserialize
    ↓
[FourierTransformer] Compute IDFT
    ↓
[EpicycleEngine] Generate Animation Frames (optional)
    ↓
[LiveRenderer] Visualize Reconstruction (optional)
    ↓
Reconstructed Image
```

## Module Structure

### Core Modules (`fourier_encryption/core/`)

- **image_processor.py**: Image loading, preprocessing, format validation
- **edge_detector.py**: Edge detection strategies (Canny, Industrial, AI)
- **contour_extractor.py**: Contour extraction and complex plane conversion
- **fourier_transformer.py**: DFT/IDFT computation, coefficient sorting
- **epicycle_engine.py**: Epicycle state computation, animation frame generation

### Encryption Modules (`fourier_encryption/encryption/`)

- **base_encryptor.py**: Abstract encryption strategy interface
- **aes_encryptor.py**: AES-256-GCM implementation with HMAC
- **key_manager.py**: Key derivation, validation, secure comparison

### AI Modules (`fourier_encryption/ai/`)

- **edge_detector.py**: CNN/ViT-based edge detection with GPU support
- **coefficient_optimizer.py**: AI-based coefficient count optimization
- **anomaly_detector.py**: Tamper detection and distribution validation
- **model_repository.py**: Model loading, versioning, metadata management

### Visualization Modules (`fourier_encryption/visualization/`)

- **live_renderer.py**: Real-time epicycle animation with Observer pattern
- **monitoring_dashboard.py**: Metrics tracking and display

### Security Modules (`fourier_encryption/security/`)

- **input_validator.py**: Input validation, injection prevention
- **path_validator.py**: Path traversal prevention, filename sanitization
- **sanitizer.py**: Sensitive data redaction for logs

### Plugin Modules (`fourier_encryption/plugins/`)

- **base_plugin.py**: Plugin interface definitions
- **plugin_registry.py**: Plugin registration and discovery
- **plugin_loader.py**: Dynamic plugin loading

### Application Module (`fourier_encryption/application/`)

- **orchestrator.py**: Main workflow coordination, dependency injection

### Configuration Modules (`fourier_encryption/config/`)

- **settings.py**: Configuration loading, validation, environment overrides
- **logging_config.py**: Structured logging with automatic sanitization
- **paths.py**: Path management for images, logs, models

### Transmission Modules (`fourier_encryption/transmission/`)

- **serializer.py**: MessagePack serialization with schema validation

### Concurrency Modules (`fourier_encryption/concurrency/`)

- **thread_pool.py**: Thread pool for parallel processing
- **concurrent_orchestrator.py**: Batch encryption with progress tracking
- **thread_safe_renderer.py**: Thread-safe visualization updates

## Component Interactions

### Encryption Workflow

```python
# 1. Initialize components with dependency injection
orchestrator = EncryptionOrchestrator(
    image_processor=ImageProcessor(),
    edge_detector=CannyEdgeDetector(),
    fourier_transformer=FourierTransformer(),
    encryptor=AES256Encryptor(),
    serializer=CoefficientSerializer()
)

# 2. Execute encryption pipeline
payload = orchestrator.encrypt_image(
    image_path=Path("image.png"),
    key="password",
    config=EncryptionConfig()
)

# 3. Save encrypted payload
with open("encrypted.json", "w") as f:
    json.dump(payload.to_dict(), f)
```

### Decryption with Visualization

```python
# 1. Load encrypted payload
with open("encrypted.json", "r") as f:
    payload_dict = json.load(f)
payload = EncryptedPayload.from_dict(payload_dict)

# 2. Decrypt and visualize
reconstructed = orchestrator.decrypt_image(
    payload=payload,
    key="password",
    visualize=True  # Enables live animation
)
```

## Extensibility Points

### 1. Custom Encryption Strategies

Implement `EncryptionStrategy` interface:

```python
class PostQuantumEncryptor(EncryptionStrategy):
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        # Implement Kyber/Dilithium encryption
        pass
    
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        # Implement decryption
        pass
```

### 2. Custom Edge Detectors

Implement `EdgeDetector` interface:

```python
class CustomEdgeDetector(EdgeDetector):
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        # Custom edge detection algorithm
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        return {"detection_time": 0.5, "quality": 0.95}
```

### 3. Custom AI Models

Use plugin system:

```python
class CustomAIPlugin(AIPlugin):
    def initialize(self, config):
        self.model = load_custom_model(config["model_path"])
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        return self.model.predict(image)
```

## Performance Considerations

### 1. Vectorization
- NumPy operations for DFT/IDFT computation
- Batch processing for multiple images
- GPU acceleration for AI models

### 2. Caching
- Frequently accessed configuration
- Loaded AI models
- Preprocessed images

### 3. Parallel Processing
- Thread pool for concurrent encryption
- Async I/O for file operations
- GPU parallelism for AI inference

### 4. Memory Management
- Streaming for large images
- Lazy loading of AI models
- Secure memory wiping for sensitive data

## Security Architecture

### Defense in Depth

1. **Input Layer**: Validation, sanitization, injection prevention
2. **Processing Layer**: Secure algorithms, constant-time operations
3. **Storage Layer**: Encrypted payloads, secure key derivation
4. **Output Layer**: Sanitized logs, redacted error messages

### Threat Model

**Protected Against:**
- Brute force attacks (PBKDF2 with 100,000+ iterations)
- Timing attacks (constant-time comparison)
- Injection attacks (input validation)
- Path traversal (path validation)
- Information leakage (automatic sanitization)
- Tampering (HMAC verification)

**Not Protected Against:**
- Physical access to decryption keys
- Side-channel attacks (power analysis, EM)
- Quantum computing (future: post-quantum algorithms)

## Testing Architecture

### Test Pyramid

```
        /\
       /  \
      / E2E \
     /--------\
    /Integration\
   /--------------\
  /   Unit Tests   \
 /------------------\
/  Property Tests    \
```

### Test Types

1. **Property-Based Tests**: Universal correctness properties (34 properties)
2. **Unit Tests**: Component-level testing (>80% coverage)
3. **Integration Tests**: End-to-end workflows
4. **Performance Tests**: Benchmarks and profiling

## Deployment Architecture

### Standalone Application

```
User → CLI → Orchestrator → Core Components → File System
```

### API Service

```
Client → REST API → Orchestrator → Core Components → Database/Storage
         ↓
    Rate Limiter
    Authentication
    CORS
```

### Distributed System (Future)

```
Load Balancer → API Servers → Message Queue → Worker Nodes
                                    ↓
                              Shared Storage
                              Model Registry
```

## Configuration Management

### Configuration Hierarchy

1. **Default Configuration**: Built-in defaults
2. **File Configuration**: YAML/JSON files
3. **Environment Variables**: Override file config
4. **Runtime Parameters**: Override all

### Configuration Validation

All configuration validated on startup:
- Type checking
- Range validation
- Required fields
- Dependency validation

## Monitoring and Observability

### Metrics Collected

- Processing time per stage
- Memory usage
- FPS for visualization
- Encryption/decryption throughput
- AI model inference time
- Error rates

### Logging Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical failures requiring immediate attention

## Future Architecture Enhancements

1. **Microservices**: Split into independent services
2. **Event-Driven**: Message queue for async processing
3. **Distributed**: Multi-node processing
4. **Cloud-Native**: Kubernetes deployment
5. **Serverless**: Lambda functions for API
6. **Blockchain**: Decentralized key management

## References

- Clean Architecture: Robert C. Martin
- SOLID Principles: Robert C. Martin
- Design Patterns: Gang of Four
- Security by Design: OWASP guidelines
- Performance Optimization: NumPy best practices

