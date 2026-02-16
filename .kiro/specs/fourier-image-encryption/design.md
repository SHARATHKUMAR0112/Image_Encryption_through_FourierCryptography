# Design Document: Fourier-Based Image Encryption System with AI Integration

## Overview

This system implements an industrial-grade image encryption mechanism using Fourier Series decomposition and epicycle-based sketch reconstruction. The core innovation lies in encrypting the Fourier coefficients that represent an image's contour, making the encrypted data mathematically meaningful yet cryptographically secure.

The system combines classical signal processing (Discrete Fourier Transform), modern cryptography (AES-256), and AI-enhanced optimization to create a unique encryption approach where:
- Images are reduced to their essential contours
- Contours are decomposed into rotating frequency components (epicycles)
- These components are encrypted and can be visually reconstructed through animation
- AI optimizes the number of coefficients needed and detects tampering

### Key Design Principles

1. **Clean Architecture**: Strict layer separation (Presentation → Application → Domain → Infrastructure)
2. **SOLID Principles**: Single responsibility, dependency inversion, interface segregation
3. **Security by Design**: Defense in depth, fail-secure defaults, constant-time operations
4. **Performance First**: Vectorized operations, GPU acceleration, parallel processing
5. **Extensibility**: Plugin architecture for algorithms, AI models, and encryption strategies

## Architecture

### Layer Structure

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

### Design Patterns

1. **Strategy Pattern**: Encryption algorithms (AES-256, future post-quantum)
2. **Factory Pattern**: Create appropriate encryptors based on configuration
3. **Observer Pattern**: Live visualization updates during epicycle animation
4. **Repository Pattern**: AI model loading and management
5. **Dependency Injection**: All components receive dependencies via constructors

### Data Flow

**Encryption Pipeline**:
```
Image → Preprocess → Edge Detection → Contour Extraction → 
Complex Plane → DFT → AI Optimization → Sort Coefficients → 
Serialize → Encrypt → HMAC → Encrypted Payload
```

**Decryption Pipeline**:
```
Encrypted Payload → Anomaly Detection → HMAC Verify → Decrypt → 
Deserialize → Coefficients → IDFT → Epicycle Animation → 
Sketch Reconstruction → Image
```

## Components and Interfaces

### 1. Image Processing Pipeline

**ImageProcessor**
```python
class ImageProcessor(ABC):
    """Abstract base for image preprocessing"""
    
    @abstractmethod
    def load_image(self, path: Path) -> np.ndarray:
        """Load image from file"""
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
        """Convert to grayscale, resize, normalize"""
        pass
    
    @abstractmethod
    def validate_format(self, path: Path) -> bool:
        """Validate image format (PNG, JPG, BMP)"""
        pass
```

**Implementation**: Uses OpenCV for image loading and preprocessing. Supports configurable resize strategies (maintain aspect ratio, pad, crop).

**EdgeDetector (Abstract Base)**
```python
class EdgeDetector(ABC):
    """Base class for edge detection strategies"""
    
    @abstractmethod
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Return binary edge map"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """Return detection time, quality metrics"""
        pass
```

**Implementations**:
- `CannyEdgeDetector`: Traditional Canny algorithm with adaptive thresholding
- `IndustrialEdgeDetector`: Production-grade pipeline with GrabCut foreground extraction, adaptive preprocessing, and morphological refinement
- `AIEdgeDetector`: CNN or Vision Transformer based, GPU-accelerated

**IndustrialEdgeDetector**
```python
@dataclass
class EdgeDetectionConfig:
    """Configuration for industrial edge detection pipeline"""
    grabcut_iterations: int = 5
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    gaussian_kernel: int = 5
    morph_kernel_size: int = 3
    enable_morphology: bool = True
    enable_foreground_extraction: bool = True

class IndustrialEdgeDetector(EdgeDetector):
    """
    Production-grade edge detection with robust foreground extraction.
    
    Features:
    - GrabCut-based foreground extraction with mask refinement
    - Adaptive Gaussian preprocessing
    - Tunable Canny edge detection
    - Morphological cleanup (closing operation)
    - Optimized for sketch-ready output
    """
    
    def __init__(self, config: EdgeDetectionConfig):
        self.config = config
    
    def extract_foreground(self, image: np.ndarray) -> np.ndarray:
        """
        Perform foreground extraction using GrabCut algorithm.
        
        Uses rectangular initialization with configurable iterations.
        Refines mask to separate foreground (GC_FGD, GC_PR_FGD) from background.
        """
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image: grayscale conversion + Gaussian blur.
        Ensures odd kernel size for Gaussian blur.
        """
        pass
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection with configurable thresholds.
        """
        pass
    
    def postprocess(self, edges: np.ndarray) -> np.ndarray:
        """
        Apply morphological refinement using elliptical kernel.
        Performs closing operation to connect nearby edges.
        """
        pass
    
    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Full pipeline: foreground extraction → preprocess → 
        edge detection → postprocess
        """
        pass
```

**ContourExtractor**
```python
@dataclass
class Contour:
    points: np.ndarray  # Nx2 array of (x, y) coordinates
    is_closed: bool
    length: int
    
class ContourExtractor:
    """Extract and process contours from edge maps"""
    
    def extract_contours(self, edge_map: np.ndarray) -> List[Contour]:
        """Find contours using OpenCV findContours"""
        pass
    
    def to_complex_plane(self, contour: Contour) -> np.ndarray:
        """Convert (x, y) points to complex numbers: x + iy"""
        pass
    
    def resample_contour(self, contour: Contour, num_points: int) -> Contour:
        """Resample to uniform point distribution"""
        pass
```

### 2. Fourier Transform Engine

**FourierCoefficient**
```python
@dataclass(frozen=True)
class FourierCoefficient:
    """Immutable Fourier coefficient representation"""
    frequency: int          # k in DFT formula
    amplitude: float        # |F(k)| - radius of epicycle
    phase: float           # arg(F(k)) - initial angle in radians
    complex_value: complex # F(k) = amplitude * e^(i*phase)
    
    def __post_init__(self):
        # Validate amplitude >= 0, phase in [-π, π]
        pass
```

**FourierTransformer**
```python
class FourierTransformer:
    """Compute DFT and IDFT on contour points"""
    
    def compute_dft(self, points: np.ndarray) -> List[FourierCoefficient]:
        """
        Compute Discrete Fourier Transform:
        F(k) = Σ(n=0 to N-1) x(n) * e^(-j*2π*k*n/N)
        
        Uses NumPy FFT for O(N log N) performance
        """
        pass
    
    def compute_idft(self, coefficients: List[FourierCoefficient]) -> np.ndarray:
        """Reconstruct points from coefficients"""
        pass
    
    def sort_by_amplitude(self, coefficients: List[FourierCoefficient]) -> List[FourierCoefficient]:
        """Sort descending by amplitude (most significant first)"""
        pass
    
    def truncate_coefficients(self, coefficients: List[FourierCoefficient], 
                             num_terms: int) -> List[FourierCoefficient]:
        """Keep only top N coefficients"""
        pass
```

**EpicycleEngine**
```python
@dataclass
class EpicycleState:
    """State at a specific time t"""
    time: float  # t in [0, 2π]
    positions: List[complex]  # Position of each epicycle center
    trace_point: complex  # Final point being drawn
    
class EpicycleEngine:
    """Compute epicycle positions for animation"""
    
    def __init__(self, coefficients: List[FourierCoefficient]):
        self.coefficients = coefficients
        
    def compute_state(self, t: float) -> EpicycleState:
        """
        Compute epicycle positions at time t:
        Each epicycle: center + radius * e^(i*(frequency*t + phase))
        """
        pass
    
    def generate_animation_frames(self, num_frames: int, 
                                  speed: float = 1.0) -> Iterator[EpicycleState]:
        """Generate frames for full rotation (t: 0 → 2π)"""
        pass
```

### 3. Encryption Layer

**EncryptionStrategy (Abstract Base)**
```python
class EncryptionStrategy(ABC):
    """Abstract encryption strategy"""
    
    @abstractmethod
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        """Encrypt data, return payload with IV and HMAC"""
        pass
    
    @abstractmethod
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        """Decrypt and verify HMAC"""
        pass
```

**AES256Encryptor**
```python
@dataclass
class EncryptedPayload:
    """Complete encrypted package"""
    ciphertext: bytes
    iv: bytes  # Initialization vector (16 bytes for AES)
    hmac: bytes  # HMAC-SHA256 (32 bytes)
    metadata: Dict[str, Any]  # Version, coefficient count, dimensions
    
class AES256Encryptor(EncryptionStrategy):
    """AES-256-GCM encryption with HMAC"""
    
    def __init__(self, kdf_iterations: int = 100_000):
        self.kdf_iterations = kdf_iterations
        
    def derive_key(self, password: str, salt: bytes) -> bytes:
        """PBKDF2-HMAC-SHA256 key derivation"""
        pass
    
    def encrypt(self, data: bytes, key: bytes) -> EncryptedPayload:
        """
        1. Generate random IV (cryptographically secure)
        2. Encrypt with AES-256-GCM
        3. Compute HMAC-SHA256 over (IV || ciphertext)
        """
        pass
    
    def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
        """
        1. Verify HMAC (constant-time comparison)
        2. Decrypt with AES-256-GCM
        3. Return plaintext or raise DecryptionError
        """
        pass
    
    def secure_wipe(self, data: bytearray):
        """Overwrite sensitive data in memory"""
        pass
```

**KeyManager**
```python
class KeyManager:
    """Secure key handling"""
    
    def generate_salt(self) -> bytes:
        """Generate 32-byte random salt"""
        pass
    
    def validate_key_strength(self, password: str) -> bool:
        """Check minimum length, entropy"""
        pass
    
    def constant_time_compare(self, a: bytes, b: bytes) -> bool:
        """Prevent timing attacks"""
        pass
```

### 4. Serialization Module

**CoefficientSerializer**
```python
class CoefficientSerializer:
    """Serialize/deserialize Fourier coefficients"""
    
    def serialize(self, coefficients: List[FourierCoefficient], 
                  metadata: Dict[str, Any]) -> bytes:
        """
        Use MessagePack for compact binary format:
        {
          "version": "1.0",
          "count": N,
          "dimensions": [width, height],
          "coefficients": [
            {"freq": k, "amp": r, "phase": θ},
            ...
          ]
        }
        """
        pass
    
    def deserialize(self, data: bytes) -> Tuple[List[FourierCoefficient], Dict]:
        """Parse and validate serialized data"""
        pass
    
    def validate_schema(self, data: Dict) -> bool:
        """Ensure all required fields present"""
        pass
```

### 5. AI Components

**AIEdgeDetector**
```python
class AIEdgeDetector(EdgeDetector):
    """CNN or Vision Transformer for edge detection"""
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.model = self.load_model(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        1. Preprocess image (normalize, resize)
        2. Run inference on GPU
        3. Post-process output (threshold, morphology)
        4. Fallback to Canny if GPU unavailable
        """
        pass
    
    def load_model(self, path: Path) -> torch.nn.Module:
        """Load pre-trained model with version check"""
        pass
```

**CoefficientOptimizer**
```python
@dataclass
class OptimizationResult:
    optimal_count: int
    complexity_class: str  # "low", "medium", "high"
    reconstruction_error: float  # RMSE
    explanation: str  # Why this count was chosen
    
class CoefficientOptimizer:
    """AI-based coefficient count optimization"""
    
    def __init__(self, model_path: Path):
        self.model = self.load_regression_model(model_path)
        
    def classify_complexity(self, image: np.ndarray) -> str:
        """Analyze image features (edges, textures, frequency content)"""
        pass
    
    def optimize_count(self, coefficients: List[FourierCoefficient], 
                      target_error: float = 0.05) -> OptimizationResult:
        """
        Binary search or RL-based selection:
        - Start with all coefficients
        - Iteratively remove low-amplitude terms
        - Stop when RMSE exceeds target_error
        """
        pass
    
    def compute_reconstruction_error(self, original: np.ndarray, 
                                    reconstructed: np.ndarray) -> float:
        """RMSE between contours"""
        pass
```

**AnomalyDetector**
```python
@dataclass
class AnomalyReport:
    is_anomalous: bool
    confidence: float
    anomaly_type: str  # "tampered", "corrupted", "none"
    severity: str  # "low", "medium", "high", "critical"
    details: str
    
class AnomalyDetector:
    """Detect tampered coefficients"""
    
    def __init__(self, model_path: Path):
        self.model = self.load_anomaly_model(model_path)
        
    def detect(self, coefficients: List[FourierCoefficient]) -> AnomalyReport:
        """
        Check for:
        1. Unusual amplitude distribution (should follow power law)
        2. Phase discontinuities
        3. Statistical outliers
        4. Frequency gaps
        """
        pass
    
    def validate_distribution(self, amplitudes: np.ndarray) -> bool:
        """Check if amplitudes follow expected decay pattern"""
        pass
```

### 6. Visualization Module

**LiveRenderer**
```python
class LiveRenderer:
    """Real-time epicycle animation using Observer pattern"""
    
    def __init__(self, backend: str = "pyqtgraph"):
        self.backend = backend  # "pyqtgraph" or "matplotlib"
        self.observers: List[RenderObserver] = []
        
    def attach_observer(self, observer: RenderObserver):
        """Register observer for frame updates"""
        pass
    
    def render_frame(self, state: EpicycleState):
        """
        Draw:
        1. All epicycle circles (decreasing opacity for smaller ones)
        2. Connecting lines between epicycle centers
        3. Trace path (accumulated points)
        4. Current drawing point (highlighted)
        """
        self.notify_observers(state)
        
    def animate(self, engine: EpicycleEngine, fps: int = 30):
        """Run animation loop, maintain target FPS"""
        pass
```

**MonitoringDashboard**
```python
@dataclass
class Metrics:
    current_coefficient_index: int
    active_radius: float
    progress_percentage: float
    fps: float
    processing_time: float
    memory_usage_mb: float
    encryption_status: str
    
class MonitoringDashboard:
    """Real-time metrics display"""
    
    def __init__(self):
        self.metrics = Metrics(...)
        self.update_thread = None
        
    def update_metrics(self, metrics: Metrics):
        """Thread-safe metrics update"""
        pass
    
    def start_monitoring(self):
        """Start background thread for metric collection"""
        pass
    
    def display(self):
        """Render dashboard (terminal or GUI)"""
        pass
```

**ImageReconstructor**
```python
@dataclass
class ReconstructionConfig:
    """Configuration for image reconstruction"""
    mode: str = "animated"  # "static" or "animated"
    speed: float = 1.0  # Animation speed multiplier (0.1x to 10x)
    quality: str = "balanced"  # "fast", "balanced", "quality"
    save_frames: bool = False  # Save individual frames
    save_animation: bool = False  # Save as video/GIF
    output_format: str = "mp4"  # "mp4", "gif", "png_sequence"
    output_path: Optional[Path] = None
    backend: str = "pyqtgraph"  # "pyqtgraph" or "matplotlib"
    
    def get_frame_count(self) -> int:
        """Return frame count based on quality mode"""
        if self.quality == "fast":
            return 30
        elif self.quality == "balanced":
            return 100
        else:  # quality
            return 300

@dataclass
class ReconstructionResult:
    """Result of reconstruction operation"""
    final_image: np.ndarray  # Final reconstructed image
    frames: Optional[List[np.ndarray]] = None  # Animation frames (if saved)
    animation_path: Optional[Path] = None  # Path to saved animation
    reconstruction_time: float = 0.0  # Time taken in seconds
    frame_count: int = 0  # Number of frames generated
    
class ImageReconstructor:
    """
    Dedicated module for reconstructing images from Fourier coefficients.
    
    Supports both static reconstruction (final image only) and animated
    reconstruction (epicycle drawing process). Integrates with existing
    epicycle_engine.py and visualization backends.
    """
    
    def __init__(self, config: ReconstructionConfig):
        self.config = config
        self.renderer = LiveRenderer(backend=config.backend)
        
    def reconstruct_static(self, coefficients: List[FourierCoefficient]) -> np.ndarray:
        """
        Perform static reconstruction - return final image only.
        
        Uses IDFT to directly compute final contour points without animation.
        Fast mode for when visualization is not needed.
        """
        pass
    
    def reconstruct_animated(self, 
                            coefficients: List[FourierCoefficient],
                            engine: EpicycleEngine) -> ReconstructionResult:
        """
        Perform animated reconstruction - generate epicycle drawing animation.
        
        Uses EpicycleEngine to compute epicycle positions at each frame.
        Renders animation using configured backend (PyQtGraph or Matplotlib).
        Optionally saves frames or complete animation.
        """
        pass
    
    def save_frames(self, frames: List[np.ndarray], output_dir: Path):
        """Save individual frames as PNG sequence"""
        pass
    
    def save_animation_video(self, frames: List[np.ndarray], output_path: Path):
        """Save frames as MP4 video using OpenCV VideoWriter"""
        pass
    
    def save_animation_gif(self, frames: List[np.ndarray], output_path: Path):
        """Save frames as animated GIF using PIL"""
        pass
    
    def render_frame_to_image(self, state: EpicycleState) -> np.ndarray:
        """
        Render a single epicycle state to an image array.
        
        Converts visualization backend output to numpy array for saving.
        """
        pass
```

### 7. Application Orchestrator

**EncryptionOrchestrator**
```python
class EncryptionOrchestrator:
    """Coordinates the entire encryption workflow"""
    
    def __init__(self, 
                 image_processor: ImageProcessor,
                 edge_detector: EdgeDetector,
                 fourier_transformer: FourierTransformer,
                 encryptor: EncryptionStrategy,
                 serializer: CoefficientSerializer,
                 optimizer: Optional[CoefficientOptimizer] = None,
                 anomaly_detector: Optional[AnomalyDetector] = None,
                 reconstructor: Optional[ImageReconstructor] = None):
        # Dependency injection
        self.image_processor = image_processor
        self.edge_detector = edge_detector
        self.fourier_transformer = fourier_transformer
        self.encryptor = encryptor
        self.serializer = serializer
        self.optimizer = optimizer
        self.anomaly_detector = anomaly_detector
        self.reconstructor = reconstructor
        
    def encrypt_image(self, image_path: Path, key: str, 
                     config: EncryptionConfig) -> EncryptedPayload:
        """
        Full encryption pipeline:
        1. Load and preprocess image
        2. Detect edges (AI or traditional)
        3. Extract contours
        4. Compute DFT
        5. Optimize coefficient count (if AI enabled)
        6. Serialize coefficients
        7. Encrypt with AES-256
        8. Return encrypted payload
        """
        pass
    
    def decrypt_image(self, payload: EncryptedPayload, key: str,
                     visualize: bool = False,
                     reconstruct: bool = False) -> np.ndarray:
        """
        Full decryption pipeline:
        1. Anomaly detection (if AI enabled)
        2. Decrypt payload
        3. Deserialize coefficients
        4. Reconstruct via IDFT or epicycle animation
        5. Optionally perform image reconstruction (if reconstruct=True)
        6. Return reconstructed image
        """
        pass
    
    def reconstruct_from_coefficients(self, 
                                     coefficients: List[FourierCoefficient],
                                     config: ReconstructionConfig) -> ReconstructionResult:
        """
        Perform image reconstruction from decrypted coefficients.
        
        This method is called after decryption to visualize the reconstruction
        process using epicycles. Can be called independently for reconstruction
        without full decryption workflow.
        """
        if not self.reconstructor:
            raise ConfigurationError("ImageReconstructor not configured")
        
        if config.mode == "static":
            image = self.reconstructor.reconstruct_static(coefficients)
            return ReconstructionResult(
                final_image=image,
                reconstruction_time=0.0,
                frame_count=1
            )
        else:  # animated
            engine = EpicycleEngine(coefficients)
            return self.reconstructor.reconstruct_animated(coefficients, engine)
```

## Data Models

### Core Data Structures

**PreprocessConfig**
```python
@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (1920, 1080)
    maintain_aspect_ratio: bool = True
    normalize: bool = True
    denoise: bool = False
    denoise_strength: float = 0.5
```

**EncryptionConfig**
```python
@dataclass
class EncryptionConfig:
    num_coefficients: Optional[int] = None  # None = auto-optimize
    use_ai_edge_detection: bool = True
    use_ai_optimization: bool = True
    use_anomaly_detection: bool = True
    kdf_iterations: int = 100_000
    visualization_enabled: bool = False
    animation_speed: float = 1.0
    reconstruction_enabled: bool = False  # Enable reconstruction after decryption
    reconstruction_config: Optional[ReconstructionConfig] = None
```

**ReconstructionConfig**
```python
@dataclass
class ReconstructionConfig:
    """Configuration for image reconstruction"""
    mode: str = "animated"  # "static" or "animated"
    speed: float = 1.0  # Animation speed multiplier (0.1x to 10x)
    quality: str = "balanced"  # "fast", "balanced", "quality"
    save_frames: bool = False  # Save individual frames
    save_animation: bool = False  # Save as video/GIF
    output_format: str = "mp4"  # "mp4", "gif", "png_sequence"
    output_path: Optional[Path] = None
    backend: str = "pyqtgraph"  # "pyqtgraph" or "matplotlib"
    
    def get_frame_count(self) -> int:
        """Return frame count based on quality mode"""
        if self.quality == "fast":
            return 30
        elif self.quality == "balanced":
            return 100
        else:  # quality
            return 300

@dataclass
class ReconstructionResult:
    """Result of reconstruction operation"""
    final_image: np.ndarray  # Final reconstructed image
    frames: Optional[List[np.ndarray]] = None  # Animation frames (if saved)
    animation_path: Optional[Path] = None  # Path to saved animation
    reconstruction_time: float = 0.0  # Time taken in seconds
    frame_count: int = 0  # Number of frames generated
```

**SystemConfig**
```python
@dataclass
class SystemConfig:
    """Loaded from YAML/JSON"""
    encryption: EncryptionConfig
    preprocessing: PreprocessConfig
    reconstruction: ReconstructionConfig  # Reconstruction settings
    ai_models: Dict[str, Path]  # {"edge_detector": "path/to/model.pt", ...}
    performance: Dict[str, Any]  # Thread counts, GPU settings
    logging: Dict[str, str]  # Levels, output paths
    
    @classmethod
    def from_file(cls, path: Path) -> "SystemConfig":
        """Load and validate configuration"""
        pass
```

### Exception Hierarchy

```python
class FourierEncryptionError(Exception):
    """Base exception"""
    pass

class ImageProcessingError(FourierEncryptionError):
    """Image loading/preprocessing failures"""
    pass

class EncryptionError(FourierEncryptionError):
    """Encryption/decryption failures"""
    pass

class DecryptionError(EncryptionError):
    """Specific to decryption (wrong key, tampered data)"""
    pass

class SerializationError(FourierEncryptionError):
    """Serialization/deserialization failures"""
    pass

class AIModelError(FourierEncryptionError):
    """AI model loading/inference failures"""
    pass

class ConfigurationError(FourierEncryptionError):
    """Invalid configuration"""
    pass
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, I identified several areas where properties can be consolidated:

1. **Round-trip properties**: DFT/IDFT, encryption/decryption, serialization/deserialization, and complex plane conversion all follow the same pattern
2. **Structural validation**: Coefficient structure, metadata presence, and configuration validation share common patterns
3. **Monotonic properties**: Progress tracking, trace accumulation, and sorting all involve monotonic relationships
4. **Error handling**: Multiple criteria about graceful failure can be combined into comprehensive error properties

### Core Properties

**Property 1: Complex Plane Round-Trip**
*For any* set of 2D contour points (x, y), converting to complex plane representation (x + iy) and back should preserve the original coordinates within floating-point precision.
**Validates: Requirements 3.1.4**

**Property 2: DFT/IDFT Round-Trip**
*For any* sequence of complex numbers representing contour points, computing the DFT and then the IDFT should reconstruct the original sequence within numerical precision (ε < 1e-10).
**Validates: Requirements 3.2.1, 3.2.6**

**Property 3: Coefficient Structure Completeness**
*For any* Fourier coefficient computed by the DFT, it must contain valid frequency (integer), amplitude (non-negative float), phase (float in [-π, π]), and complex value fields, where amplitude and phase are consistent with the complex value.
**Validates: Requirements 3.2.2**

**Property 4: Amplitude Sorting Invariant**
*For any* list of Fourier coefficients after sorting by amplitude, each coefficient's amplitude must be greater than or equal to the next coefficient's amplitude (monotonically decreasing).
**Validates: Requirements 3.2.3**

**Property 5: Coefficient Count Validation**
*For any* requested number of Fourier terms N, if N is in the valid range [10, 1000], the system should accept it; otherwise, it should reject with a validation error.
**Validates: Requirements 3.2.4**

**Property 6: Epicycle Radius Consistency**
*For any* epicycle state computed from Fourier coefficients, the radius of each epicycle must exactly equal the amplitude of its corresponding coefficient.
**Validates: Requirements 3.3.2**

**Property 7: Animation Speed Bounds**
*For any* animation speed setting S, if S is in the valid range [0.1, 10.0], the system should accept it and produce frame timing consistent with that speed; otherwise, it should reject with a validation error.
**Validates: Requirements 3.3.3**

**Property 8: Trace Path Monotonic Growth**
*For any* animation sequence, the number of points in the trace path must increase monotonically (never decrease) as the animation progresses from t=0 to t=2π.
**Validates: Requirements 3.3.4, 3.6.2**

**Property 9: IV Uniqueness**
*For any* two independent encryption operations (even with the same input and key), the generated initialization vectors (IVs) must be different with overwhelming probability (collision probability < 2^-128).
**Validates: Requirements 3.4.3**

**Property 10: HMAC Integrity Validation**
*For any* encrypted payload, if the ciphertext or IV is modified by even a single bit, HMAC verification must fail and decryption must raise an integrity error.
**Validates: Requirements 3.4.4**

**Property 11: Encryption Round-Trip**
*For any* list of Fourier coefficients and encryption key, encrypting then decrypting with the same key should recover the original coefficients exactly (all fields: frequency, amplitude, phase).
**Validates: Requirements 3.4.5**

**Property 12: Wrong Key Rejection**
*For any* encrypted payload and incorrect decryption key, the decryption operation must fail gracefully with a DecryptionError exception (not crash or return corrupted data).
**Validates: Requirements 3.4.6**

**Property 13: Serialization Round-Trip**
*For any* list of Fourier coefficients with metadata, serializing then deserializing should recover the original coefficients and metadata within floating-point precision.
**Validates: Requirements 3.5.1, 3.5.2, 3.5.4**

**Property 14: Corruption Detection**
*For any* serialized data that is corrupted (random bit flips), deserialization must detect the corruption and raise a SerializationError (not silently return invalid data).
**Validates: Requirements 3.5.3**

**Property 15: Progress Percentage Bounds**
*For any* processing state during encryption or decryption, the progress percentage must be in the range [0, 100] and must increase monotonically over time.
**Validates: Requirements 3.6.4, 3.7.3**

**Property 16: Metrics Validity**
*For any* monitoring dashboard state, all displayed metrics (FPS, processing time, memory usage) must be non-negative, and FPS must be ≤ the configured maximum frame rate.
**Validates: Requirements 3.7.5**

**Property 17: Complexity Classification Validity**
*For any* input image, the AI complexity classifier must return exactly one of the valid complexity classes: "low", "medium", or "high".
**Validates: Requirements 3.9.1**

**Property 18: Optimizer Coefficient Count Bounds**
*For any* image processed by the AI optimizer, the returned optimal coefficient count must be within the valid range [10, 1000].
**Validates: Requirements 3.9.2**

**Property 19: Reconstruction Error Threshold**
*For any* optimized coefficient set, the reconstruction error (RMSE) between the original contour and the reconstructed contour must be below 5%.
**Validates: Requirements 3.9.3**

**Property 20: Optimization Reduces Size**
*For any* image processed by the AI optimizer, the optimized coefficient count must be less than or equal to the maximum coefficient count (optimization never increases size).
**Validates: Requirements 3.9.4**

**Property 21: Optimization Explanation Presence**
*For any* optimization result, the explanation field must be a non-empty string describing why the specific coefficient count was chosen.
**Validates: Requirements 3.9.6**

**Property 22: Tampered Payload Detection**
*For any* encrypted payload where coefficients have been tampered with (modified after encryption), the anomaly detector must flag it as anomalous before decryption is attempted.
**Validates: Requirements 3.10.1**

**Property 23: Coefficient Distribution Validation**
*For any* valid set of Fourier coefficients, the amplitude distribution must approximately follow a power-law decay (each amplitude ≤ previous amplitude); invalid distributions must fail validation.
**Validates: Requirements 3.10.3**

**Property 24: Anomaly Severity Validity**
*For any* anomaly detection result, the severity level must be one of: "low", "medium", "high", or "critical".
**Validates: Requirements 3.10.5**

**Property 25: Exception Type Correctness**
*For any* error condition in the system, the raised exception must be a subclass of FourierEncryptionError and must be the most specific exception type for that error (e.g., DecryptionError for decryption failures, not generic Exception).
**Validates: Requirements 3.11.8**

**Property 26: Configuration Completeness**
*For any* loaded configuration, all required sections (encryption, preprocessing, ai_models, performance, logging) must be present with valid values, or a ConfigurationError must be raised.
**Validates: Requirements 3.14.2, 3.14.3**

**Property 27: API Response Structure**
*For any* API request to /encrypt, /decrypt, or /visualize endpoints, the response must be valid JSON with an appropriate HTTP status code (2xx for success, 4xx for client errors, 5xx for server errors).
**Validates: Requirements 3.16.4**

**Property 28: Key Material Never Logged**
*For any* operation involving encryption keys, the key material must never appear in log output, error messages, or exception details (only key identifiers or hashes may be logged).
**Validates: Requirements 3.18.1**

**Property 29: Input Validation Prevents Injection**
*For any* user-provided input (file paths, keys, configuration values), the system must sanitize or reject inputs containing potentially malicious patterns (path traversal, command injection, SQL injection).
**Validates: Requirements 3.18.3**

**Property 30: Model Version Metadata**
*For any* loaded AI model, the model metadata must include a valid version string (semantic versioning format: major.minor.patch).
**Validates: Requirements 3.19.3**

### Edge Case Properties

These properties focus on boundary conditions and special cases:

**Property 31: Empty Image Handling**
*For any* image with no detectable edges (completely uniform), the system must handle it gracefully by either returning an empty coefficient list or raising an ImageProcessingError with a descriptive message.
**Validates: Requirements 3.1.3**

**Property 32: Minimum Coefficient Count**
*For any* contour with fewer than 10 points, the system must either interpolate to reach the minimum or raise a validation error (never produce fewer than 10 coefficients if processing succeeds).
**Validates: Requirements 3.2.4**

**Property 33: GPU Fallback**
*For any* AI operation when GPU is unavailable or fails, the system must automatically fall back to CPU-based processing without raising an error.
**Validates: Requirements 3.8.5**

**Property 34: Concurrent Encryption Safety**
*For any* two concurrent encryption operations on different images, both operations must complete successfully without data corruption or race conditions.
**Validates: Requirements 3.13.3**

### Reconstruction Module Properties

**Property 35: Reconstructor Accepts Valid Coefficients**
*For any* valid list of Fourier coefficients, the ImageReconstructor must accept them as input without raising an error and produce a valid reconstruction result.
**Validates: Requirements 3.21.1**

**Property 36: Static Mode Returns Image Without Animation**
*For any* valid list of Fourier coefficients, when reconstruction mode is "static", the reconstructor must return a final image without generating animation frames (frame count = 1).
**Validates: Requirements 3.21.2**

**Property 37: Animated Mode Generates Frames**
*For any* valid list of Fourier coefficients, when reconstruction mode is "animated", the reconstructor must generate multiple frames (frame count > 1) and return a ReconstructionResult with frames.
**Validates: Requirements 3.21.3**

**Property 38: Epicycle Engine Integration Consistency**
*For any* valid list of Fourier coefficients, the ImageReconstructor using EpicycleEngine must produce the same final contour points as direct IDFT computation (within numerical precision).
**Validates: Requirements 3.21.4**

**Property 39: Backend Compatibility**
*For any* valid list of Fourier coefficients and backend choice ("pyqtgraph" or "matplotlib"), the ImageReconstructor must complete reconstruction successfully without errors.
**Validates: Requirements 3.21.5**

**Property 40: Output Format Support**
*For any* valid reconstruction result with frames, saving in any supported format ("mp4", "gif", "png_sequence") must succeed and produce a valid output file.
**Validates: Requirements 3.21.6, 3.21.7**

**Property 41: Reconstruction Speed Bounds**
*For any* reconstruction speed S, if S is in the valid range [0.1, 10.0], the system must accept it; otherwise, it must reject with a validation error.
**Validates: Requirements 3.21.8**

**Property 42: Conditional Reconstruction Based on Config**
*For any* decryption operation, if reconstruction_enabled is True in the configuration, reconstruction must be performed; if False, reconstruction must be skipped.
**Validates: Requirements 3.22.1**

**Property 43: Frame Streaming Progressiveness**
*For any* animated reconstruction, frames must be generated and streamed progressively (frame N+1 is generated after frame N), not all at once.
**Validates: Requirements 3.22.6**

**Property 44: Reconstruction Configuration Completeness**
*For any* loaded reconstruction configuration, all required fields (mode, speed, quality, output_format, backend) must be present with valid values, or a ConfigurationError must be raised.
**Validates: Requirements 3.23.1, 3.23.2, 3.23.3**

**Property 45: Quality Mode Validation**
*For any* reconstruction quality mode Q, Q must be exactly one of the valid modes: "fast", "balanced", or "quality"; otherwise, a validation error must be raised.
**Validates: Requirements 3.23.4**

**Property 46: Quality Mode Frame Counts**
*For any* reconstruction operation, the frame count must match the quality mode: fast mode must generate exactly 30 frames, balanced mode must generate exactly 100 frames, and quality mode must generate exactly 300 frames.
**Validates: Requirements 3.23.5, 3.23.6**

## Error Handling

### Error Handling Strategy

The system implements a defense-in-depth error handling approach with multiple layers:

1. **Input Validation Layer**: Validate all inputs at system boundaries (CLI, API, file loading)
2. **Domain Validation Layer**: Enforce business rules and mathematical constraints
3. **Infrastructure Error Handling**: Handle external failures (file I/O, network, GPU)
4. **Graceful Degradation**: Fall back to simpler methods when advanced features fail

### Exception Hierarchy and Usage

```python
FourierEncryptionError (base)
├── ImageProcessingError
│   ├── InvalidFormatError (unsupported image format)
│   ├── EdgeDetectionError (edge detection failed)
│   └── ContourExtractionError (no valid contours found)
├── EncryptionError
│   ├── DecryptionError (wrong key, tampered data)
│   ├── KeyDerivationError (KDF failed)
│   └── IntegrityError (HMAC validation failed)
├── SerializationError
│   ├── DeserializationError (corrupted data)
│   └── SchemaValidationError (missing required fields)
├── AIModelError
│   ├── ModelLoadError (model file not found or corrupted)
│   ├── InferenceError (model inference failed)
│   └── GPUError (GPU operation failed)
└── ConfigurationError
    ├── InvalidConfigError (malformed config file)
    └── ValidationError (config values out of range)
```

### Error Handling Patterns

**Pattern 1: Fail-Fast Validation**
```python
def encrypt_image(self, image_path: Path, key: str) -> EncryptedPayload:
    # Validate inputs immediately
    if not image_path.exists():
        raise ImageProcessingError(f"Image not found: {image_path}")
    
    if not self.key_manager.validate_key_strength(key):
        raise EncryptionError("Key does not meet minimum strength requirements")
    
    # Proceed with operation...
```

**Pattern 2: Graceful Degradation**
```python
def detect_edges(self, image: np.ndarray) -> np.ndarray:
    try:
        # Try AI-based detection first
        return self.ai_detector.detect_edges(image)
    except (AIModelError, GPUError) as e:
        logger.warning(f"AI detection failed: {e}, falling back to Canny")
        return self.canny_detector.detect_edges(image)
```

**Pattern 3: Resource Cleanup**
```python
def decrypt(self, payload: EncryptedPayload, key: bytes) -> bytes:
    key_material = None
    try:
        key_material = self.derive_key(key, payload.salt)
        plaintext = self._decrypt_aes(payload.ciphertext, key_material, payload.iv)
        return plaintext
    finally:
        # Always wipe sensitive data
        if key_material:
            self.secure_wipe(key_material)
```

**Pattern 4: Context-Rich Errors**
```python
def deserialize(self, data: bytes) -> List[FourierCoefficient]:
    try:
        parsed = msgpack.unpackb(data)
    except Exception as e:
        raise SerializationError(
            f"Failed to deserialize coefficient data: {e}",
            context={"data_length": len(data), "first_bytes": data[:16].hex()}
        )
```

### Logging Strategy

**Log Levels**:
- **DEBUG**: Detailed diagnostic information (coefficient values, intermediate results)
- **INFO**: Normal operation milestones (encryption started, optimization complete)
- **WARNING**: Recoverable issues (GPU unavailable, falling back to CPU)
- **ERROR**: Operation failures (decryption failed, invalid configuration)
- **CRITICAL**: System-level failures (unable to load required models)

**Structured Logging Format**:
```python
logger.info(
    "Encryption completed",
    extra={
        "operation": "encrypt",
        "image_path": str(image_path),
        "coefficient_count": len(coefficients),
        "processing_time_ms": elapsed_ms,
        "optimization_enabled": config.use_ai_optimization
    }
)
```

**Security Considerations**:
- Never log encryption keys, passwords, or sensitive data
- Log only key identifiers (hash of key) for debugging
- Sanitize file paths to prevent information disclosure
- Use constant-time operations for key comparison

## Testing Strategy

### Dual Testing Approach

The system requires both unit tests and property-based tests for comprehensive coverage:

**Unit Tests**: Focus on specific examples, edge cases, and integration points
**Property Tests**: Verify universal properties across all inputs through randomization

### Property-Based Testing Configuration

**Framework**: Use `hypothesis` library for Python property-based testing

**Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Each test tagged with: `# Feature: fourier-image-encryption, Property N: [property text]`
- Seed-based reproducibility for failed tests
- Shrinking enabled to find minimal failing examples

**Example Property Test Structure**:
```python
from hypothesis import given, strategies as st
import pytest

@given(
    points=st.lists(
        st.tuples(st.floats(min_value=-1000, max_value=1000), 
                  st.floats(min_value=-1000, max_value=1000)),
        min_size=10,
        max_size=1000
    )
)
@pytest.mark.property_test
def test_complex_plane_round_trip(points):
    """
    Feature: fourier-image-encryption
    Property 1: Complex Plane Round-Trip
    
    For any set of 2D points, converting to complex plane and back
    should preserve the original coordinates.
    """
    contour = Contour(points=np.array(points), is_closed=True, length=len(points))
    extractor = ContourExtractor()
    
    # Convert to complex plane
    complex_points = extractor.to_complex_plane(contour)
    
    # Convert back to 2D
    recovered_points = np.column_stack([complex_points.real, complex_points.imag])
    
    # Verify round-trip within floating-point precision
    np.testing.assert_allclose(recovered_points, points, rtol=1e-10)
```

### Unit Testing Strategy

**Test Organization**:
```
tests/
├── unit/
│   ├── test_image_processor.py
│   ├── test_fourier_transformer.py
│   ├── test_encryption.py
│   ├── test_serialization.py
│   ├── test_ai_models.py
│   └── test_visualization.py
├── integration/
│   ├── test_encryption_workflow.py
│   ├── test_decryption_workflow.py
│   └── test_api_endpoints.py
├── property/
│   ├── test_properties_core.py
│   ├── test_properties_encryption.py
│   └── test_properties_ai.py
└── fixtures/
    ├── sample_images/
    ├── test_configs/
    └── mock_models/
```

**Unit Test Focus Areas**:

1. **Specific Examples**: Test known inputs with expected outputs
   - Example: Encrypt a specific image, verify ciphertext structure
   - Example: Parse a known configuration file, verify loaded values

2. **Edge Cases**: Test boundary conditions
   - Empty images, single-pixel images
   - Minimum/maximum coefficient counts
   - Invalid file formats, corrupted data

3. **Error Conditions**: Test failure modes
   - Wrong decryption key
   - Corrupted encrypted payload
   - Missing configuration fields
   - GPU unavailable scenarios

4. **Integration Points**: Test component interactions
   - Full encryption pipeline (image → encrypted payload)
   - Full decryption pipeline (encrypted payload → reconstructed image)
   - API request/response cycles

**Example Unit Test**:
```python
def test_wrong_key_raises_decryption_error():
    """Test that decryption with wrong key fails gracefully"""
    # Arrange
    encryptor = AES256Encryptor()
    original_data = b"test data"
    correct_key = b"correct_key_32_bytes_long_here!"
    wrong_key = b"wrong_key_32_bytes_long_here!!!"
    
    # Act
    payload = encryptor.encrypt(original_data, correct_key)
    
    # Assert
    with pytest.raises(DecryptionError) as exc_info:
        encryptor.decrypt(payload, wrong_key)
    
    assert "HMAC verification failed" in str(exc_info.value)
```

### Integration Testing

**End-to-End Workflows**:
1. **Encryption Workflow**: Image file → Encrypted payload file
2. **Decryption Workflow**: Encrypted payload file → Reconstructed image
3. **Visualization Workflow**: Encrypted payload → Animated reconstruction
4. **API Workflow**: HTTP request → JSON response

**Test Data**:
- Sample images: simple shapes, complex drawings, photographs
- Pre-encrypted payloads with known keys
- Configuration files for different scenarios

### Performance Testing

**Benchmarks** (not part of unit tests, separate benchmark suite):
- Image processing time vs. resolution
- DFT computation time vs. number of points
- Encryption time vs. coefficient count
- AI model inference time (GPU vs. CPU)
- Memory usage profiling

**Performance Assertions** (in integration tests):
- Verify operations complete within reasonable time bounds
- Verify memory usage stays within acceptable limits
- Not strict timing requirements (those are benchmarks)

### Test Coverage Goals

- **Unit Test Coverage**: >80% line coverage for core modules
- **Property Test Coverage**: All 34 correctness properties implemented
- **Integration Test Coverage**: All major workflows (encrypt, decrypt, visualize)
- **Edge Case Coverage**: All identified edge cases tested

### Continuous Testing

**Pre-commit Hooks**:
- Run fast unit tests (<5 seconds)
- Run linters (flake8, mypy)
- Check code formatting (black)

**CI Pipeline**:
- Run full unit test suite
- Run property tests (100 iterations each)
- Run integration tests
- Generate coverage reports
- Run security scans (bandit)

### Test Fixtures and Mocks

**Fixtures**:
- Sample images (various formats and sizes)
- Pre-computed Fourier coefficients
- Mock AI models (lightweight, deterministic)
- Test configuration files

**Mocking Strategy**:
- Mock external dependencies (file I/O, network)
- Mock GPU operations for CPU-only testing
- Mock AI models for fast unit tests
- Use real implementations for integration tests

### Testing AI Components

**AI Model Testing Challenges**:
- Non-deterministic outputs
- Require large test datasets
- GPU dependencies

**AI Testing Strategy**:
1. **Unit Tests**: Test model loading, input/output shapes, error handling
2. **Property Tests**: Test that outputs are in valid ranges, have correct structure
3. **Benchmark Tests**: Measure accuracy on held-out test set (not unit tests)
4. **Mock Models**: Use simple deterministic models for fast testing

**Example AI Property Test**:
```python
@given(image=st.arrays(dtype=np.uint8, shape=(256, 256)))
def test_ai_edge_detector_output_shape(image):
    """
    Feature: fourier-image-encryption
    Property: AI edge detector output has correct shape
    
    For any input image, the edge detector output should have the same
    spatial dimensions as the input.
    """
    detector = AIEdgeDetector(model_path=MOCK_MODEL_PATH)
    edges = detector.detect_edges(image)
    
    assert edges.shape == image.shape
    assert edges.dtype == np.uint8
    assert np.all((edges == 0) | (edges == 255))  # Binary edge map
```

### Test Documentation

Each test should include:
- Clear docstring explaining what is being tested
- Reference to requirements (for traceability)
- Reference to property number (for property tests)
- Explanation of test setup and assertions

This comprehensive testing strategy ensures that the Fourier-Based Image Encryption System is correct, reliable, and maintainable.
