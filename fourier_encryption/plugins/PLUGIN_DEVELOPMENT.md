# Plugin Development Guide

This guide explains how to develop custom plugins for the Fourier-Based Image Encryption System.

## Overview

The plugin system enables extensibility through:
- **Encryption Plugins**: Custom encryption algorithms (including post-quantum)
- **AI Model Plugins**: Custom AI models for edge detection, optimization, and anomaly detection

## Plugin Architecture

### Plugin Lifecycle

1. **Discovery**: Plugin loader scans plugin directories
2. **Loading**: Plugin module is dynamically imported
3. **Registration**: Plugin is registered with appropriate registry
4. **Initialization**: Plugin resources are set up with configuration
5. **Usage**: Plugin is used by the system
6. **Cleanup**: Plugin resources are released

### Plugin Locations

Plugins are automatically discovered from:
- **User plugins**: `~/.fourier_encryption/plugins/`
- **System plugins** (Linux/Mac): `/etc/fourier_encryption/plugins/`
- **System plugins** (Windows): `%APPDATA%/fourier_encryption/plugins/`
- **Custom directories**: Specified programmatically

## Creating an Encryption Plugin

### Basic Structure

```python
from pathlib import Path
from typing import Dict, Any
from fourier_encryption.plugins import EncryptionPlugin, PluginMetadata
from fourier_encryption.models.encryption_payload import EncryptedPayload

class MyCustomEncryptor(EncryptionPlugin):
    """Custom encryption algorithm plugin."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-custom-encryptor",
            version="1.0.0",
            author="Your Name",
            description="Custom encryption algorithm",
            plugin_type="encryption",
            dependencies={"cryptography": ">=41.0.0"}
        )
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        self.key_size = config.get("key_size", 256)
        # Set up any resources needed
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        # Release any resources
        pass
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        # Implement key derivation (e.g., PBKDF2, Argon2)
        pass
    
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        """Encrypt data."""
        # Implement encryption algorithm
        # Return EncryptedPayload with ciphertext, IV, HMAC, metadata
        pass
    
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        """Decrypt data."""
        # Implement decryption algorithm
        # Verify HMAC before decrypting
        # Return plaintext or raise DecryptionError
        pass
```

### Post-Quantum Encryption Example

```python
from fourier_encryption.plugins import EncryptionPlugin, PluginMetadata
from fourier_encryption.models.encryption_payload import EncryptedPayload
import oqs  # liboqs-python for post-quantum crypto

class KyberEncryptor(EncryptionPlugin):
    """
    Post-quantum encryption using Kyber algorithm.
    
    Kyber is a lattice-based key encapsulation mechanism (KEM)
    selected by NIST for post-quantum standardization.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="kyber-encryptor",
            version="1.0.0",
            author="Security Team",
            description="Post-quantum encryption using Kyber KEM",
            plugin_type="encryption",
            dependencies={"liboqs-python": ">=0.8.0"}
        )
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize Kyber KEM."""
        self.kem_algorithm = config.get("kem_algorithm", "Kyber768")
        self.kem = oqs.KeyEncapsulation(self.kem_algorithm)
    
    def cleanup(self) -> None:
        """Clean up KEM resources."""
        if hasattr(self, 'kem'):
            del self.kem
    
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive key using Argon2."""
        from argon2 import PasswordHasher
        ph = PasswordHasher()
        return ph.hash(password, salt=salt)
    
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        """
        Encrypt using Kyber KEM + AES-GCM.
        
        1. Generate Kyber keypair
        2. Encapsulate shared secret
        3. Use shared secret for AES-GCM encryption
        """
        # Generate keypair
        public_key = self.kem.generate_keypair()
        
        # Encapsulate to get shared secret
        ciphertext_kem, shared_secret = self.kem.encap_secret(public_key)
        
        # Use shared secret for symmetric encryption
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(shared_secret[:32])  # Use first 32 bytes
        
        import os
        iv = os.urandom(12)
        ciphertext = aesgcm.encrypt(iv, data, None)
        
        # Compute HMAC
        import hmac
        import hashlib
        hmac_value = hmac.new(key, ciphertext_kem + iv + ciphertext, hashlib.sha256).digest()
        
        return EncryptedPayload(
            ciphertext=ciphertext,
            iv=iv,
            hmac=hmac_value,
            metadata={
                "algorithm": "Kyber-AES-GCM",
                "kem_ciphertext": ciphertext_kem.hex(),
                "public_key": public_key.hex(),
            }
        )
    
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        """Decrypt using Kyber KEM + AES-GCM."""
        # Verify HMAC first
        import hmac
        import hashlib
        kem_ciphertext = bytes.fromhex(payload.metadata["kem_ciphertext"])
        expected_hmac = hmac.new(
            key,
            kem_ciphertext + payload.iv + payload.ciphertext,
            hashlib.sha256
        ).digest()
        
        if not hmac.compare_digest(expected_hmac, payload.hmac):
            raise DecryptionError("HMAC verification failed")
        
        # Decapsulate shared secret
        shared_secret = self.kem.decap_secret(kem_ciphertext)
        
        # Decrypt with AES-GCM
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(shared_secret[:32])
        plaintext = aesgcm.decrypt(payload.iv, payload.ciphertext, None)
        
        return plaintext
```

## Creating an AI Model Plugin

### Basic Structure

```python
from pathlib import Path
from typing import Dict, Any
import numpy as np
from fourier_encryption.plugins import AIModelPlugin, PluginMetadata

class MyCustomEdgeDetector(AIModelPlugin):
    """Custom AI edge detection model."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-edge-detector",
            version="1.0.0",
            author="Your Name",
            description="Custom edge detection model",
            plugin_type="ai_model",
            dependencies={"torch": ">=2.0.0"}
        )
    
    @property
    def model_type(self) -> str:
        """Return model type."""
        return "edge_detector"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize model."""
        model_path = Path(config["model_path"])
        self.load_model(model_path)
        self.device = config.get("device", "cuda")
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if hasattr(self, 'model'):
            del self.model
    
    def load_model(self, model_path: Path) -> None:
        """Load model from disk."""
        import torch
        self.model = torch.load(model_path)
        self.model.eval()
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run edge detection inference."""
        import torch
        
        # Preprocess
        tensor = torch.from_numpy(input_data).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Inference
        with torch.no_grad():
            output = self.model(tensor)
        
        # Postprocess
        edges = output.squeeze().cpu().numpy()
        edges = (edges > 0.5).astype(np.uint8) * 255
        
        return edges
    
    def get_input_shape(self) -> tuple:
        """Return expected input shape."""
        return (256, 256)
    
    def get_output_shape(self) -> tuple:
        """Return expected output shape."""
        return (256, 256)
    
    @property
    def supports_gpu(self) -> bool:
        """Check GPU support."""
        import torch
        return torch.cuda.is_available()
```

### Vision Transformer Example

```python
from fourier_encryption.plugins import AIModelPlugin, PluginMetadata
import numpy as np
from pathlib import Path

class ViTEdgeDetector(AIModelPlugin):
    """
    Vision Transformer for edge detection.
    
    Uses a pre-trained ViT model fine-tuned for edge detection.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="vit-edge-detector",
            version="1.0.0",
            author="AI Team",
            description="Vision Transformer edge detection",
            plugin_type="ai_model",
            dependencies={
                "torch": ">=2.0.0",
                "transformers": ">=4.30.0"
            }
        )
    
    @property
    def model_type(self) -> str:
        return "edge_detector"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize ViT model."""
        from transformers import ViTForImageClassification, ViTImageProcessor
        
        model_name = config.get("model_name", "google/vit-base-patch16-224")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        model_path = Path(config["model_path"])
        self.load_model(model_path)
        
        self.device = config.get("device", "cuda")
        if self.supports_gpu:
            self.model = self.model.to(self.device)
    
    def cleanup(self) -> None:
        """Clean up model."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
    
    def load_model(self, model_path: Path) -> None:
        """Load fine-tuned ViT model."""
        from transformers import ViTForImageClassification
        self.model = ViTForImageClassification.from_pretrained(str(model_path))
        self.model.eval()
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run ViT inference for edge detection."""
        import torch
        from PIL import Image
        
        # Convert to PIL Image
        if input_data.ndim == 2:
            image = Image.fromarray(input_data, mode='L')
        else:
            image = Image.fromarray(input_data)
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        if self.supports_gpu:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Postprocess to edge map
        edge_map = torch.sigmoid(logits).squeeze().cpu().numpy()
        edge_map = (edge_map > 0.5).astype(np.uint8) * 255
        
        return edge_map
    
    def get_input_shape(self) -> tuple:
        return (224, 224)
    
    def get_output_shape(self) -> tuple:
        return (224, 224)
    
    @property
    def supports_gpu(self) -> bool:
        import torch
        return torch.cuda.is_available()
```

## Using Plugins

### Programmatic Usage

```python
from fourier_encryption.plugins import PluginLoader

# Create loader
loader = PluginLoader()

# Load from standard locations
loader.load_from_standard_locations(auto_initialize=True)

# Or load from custom directory
from pathlib import Path
custom_dir = Path("/path/to/custom/plugins")
loader.load_and_register(custom_dir, auto_initialize=True)

# Get encryption plugin
encryption_registry = loader.encryption_registry
my_encryptor = encryption_registry.get_encryptor("my-custom-encryptor")

# Use plugin
encrypted = my_encryptor.encrypt(b"data", b"key")
decrypted = my_encryptor.decrypt(encrypted, b"key")

# Get AI model plugin
ai_registry = loader.ai_model_registry
my_detector = ai_registry.get_model("my-edge-detector")

# Use plugin
edges = my_detector.predict(image_array)

# Cleanup when done
loader.cleanup()
```

### Configuration File

Add plugin configuration to `config.yaml`:

```yaml
plugins:
  encryption:
    - name: "kyber-encryptor"
      enabled: true
      config:
        kem_algorithm: "Kyber768"
        
  ai_models:
    - name: "vit-edge-detector"
      enabled: true
      config:
        model_path: "/path/to/vit_edge_model"
        device: "cuda"
```

## Extension Points

### 1. Encryption Strategies

**Use Cases**:
- Post-quantum encryption (Kyber, Dilithium, SPHINCS+)
- Hardware-accelerated encryption (Intel AES-NI, ARM Crypto Extensions)
- Custom encryption schemes for research

**Requirements**:
- Implement `EncryptionPlugin` interface
- Provide key derivation, encryption, and decryption
- Include HMAC or equivalent integrity protection
- Handle errors gracefully

### 2. AI Edge Detection Models

**Use Cases**:
- Custom CNN architectures
- Vision Transformers (ViT, Swin Transformer)
- Hybrid models (CNN + Transformer)
- Domain-specific models (medical imaging, satellite imagery)

**Requirements**:
- Implement `AIModelPlugin` interface with `model_type="edge_detector"`
- Accept grayscale or RGB images
- Return binary edge map (0 or 255)
- Support GPU acceleration when available

### 3. AI Coefficient Optimizers

**Use Cases**:
- Reinforcement learning agents
- Neural architecture search
- Custom optimization algorithms

**Requirements**:
- Implement `AIModelPlugin` interface with `model_type="optimizer"`
- Accept image features as input
- Return optimal coefficient count
- Provide explainability

### 4. AI Anomaly Detectors

**Use Cases**:
- Deep learning anomaly detection
- Statistical models
- Ensemble methods

**Requirements**:
- Implement `AIModelPlugin` interface with `model_type="anomaly_detector"`
- Accept coefficient arrays as input
- Return anomaly score and classification
- Detect tampering and corruption

## Best Practices

### Security

1. **Validate all inputs**: Check data types, ranges, and formats
2. **Use constant-time operations**: Prevent timing attacks
3. **Secure key handling**: Never log keys, wipe from memory after use
4. **Implement proper HMAC**: Verify integrity before decryption
5. **Handle errors securely**: Don't leak sensitive information in errors

### Performance

1. **Use vectorized operations**: NumPy, PyTorch for batch processing
2. **GPU acceleration**: Leverage CUDA when available
3. **Lazy loading**: Load models only when needed
4. **Resource cleanup**: Release GPU memory, close files
5. **Caching**: Cache frequently used computations

### Compatibility

1. **Version dependencies**: Specify exact version requirements
2. **Backward compatibility**: Support older data formats
3. **Platform support**: Test on Linux, Mac, Windows
4. **Python versions**: Support Python 3.9+

### Testing

1. **Unit tests**: Test each method independently
2. **Integration tests**: Test plugin with system
3. **Property tests**: Verify correctness properties
4. **Performance tests**: Benchmark speed and memory

## Troubleshooting

### Plugin Not Discovered

- Check plugin file is in correct directory
- Verify file has `.py` extension
- Ensure file doesn't start with underscore
- Check file permissions

### Plugin Registration Failed

- Verify plugin inherits from correct base class
- Check all abstract methods are implemented
- Validate metadata fields
- Review error logs

### Plugin Initialization Failed

- Check configuration is valid
- Verify dependencies are installed
- Check file paths exist
- Review initialization code

### Plugin Performance Issues

- Profile code to find bottlenecks
- Use GPU acceleration when available
- Optimize data preprocessing
- Consider caching

## Future Research Directions

The plugin system enables research in:

1. **Post-Quantum Cryptography**: Kyber, Dilithium, SPHINCS+, NTRU
2. **Homomorphic Encryption**: Compute on encrypted data
3. **Zero-Knowledge Proofs**: Prove properties without revealing data
4. **Federated Learning**: Train models on distributed data
5. **Neural Compression**: Learn optimal representations
6. **Adversarial Robustness**: Defend against attacks
7. **Explainable AI**: Interpret model decisions
8. **Hardware Acceleration**: FPGA, ASIC, quantum computing

## Support

For questions or issues:
- GitHub Issues: [repository URL]
- Documentation: [docs URL]
- Email: [support email]
