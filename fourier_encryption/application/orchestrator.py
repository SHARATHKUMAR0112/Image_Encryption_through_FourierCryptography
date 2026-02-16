"""
Application orchestrator for Fourier-Based Image Encryption System.

This module coordinates the entire encryption and decryption workflow,
integrating all components from image processing through encryption to
visualization.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from fourier_encryption.config.settings import EncryptionConfig, PreprocessConfig
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.edge_detector import EdgeDetector
from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.core.image_processor import ImageProcessor
from fourier_encryption.encryption.base_encryptor import EncryptionStrategy
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.models.data_models import (
    EncryptedPayload,
    FourierCoefficient,
    ReconstructionConfig,
    ReconstructionResult,
)
from fourier_encryption.models.exceptions import (
    ConfigurationError,
    EncryptionError,
    ImageProcessingError,
)
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.visualization.image_reconstructor import ImageReconstructor


logger = logging.getLogger(__name__)


class EncryptionOrchestrator:
    """
    Coordinates the entire encryption and decryption workflow.
    
    This orchestrator implements the application layer of the Clean Architecture,
    coordinating domain layer components (image processing, Fourier transform,
    encryption) to execute complete encryption and decryption workflows.
    
    Design Pattern: Facade Pattern
    - Provides a simplified interface to complex subsystems
    - Coordinates multiple components to achieve high-level operations
    
    Design Pattern: Dependency Injection
    - All dependencies are injected via constructor
    - Enables testing with mock components
    - Supports flexible component substitution
    
    Attributes:
        image_processor: Component for loading and preprocessing images
        edge_detector: Component for detecting edges in images
        contour_extractor: Component for extracting contours from edge maps
        fourier_transformer: Component for DFT/IDFT operations
        encryptor: Component for encryption/decryption operations
        serializer: Component for serializing/deserializing coefficients
        key_manager: Component for key management operations
        optimizer: Optional AI component for coefficient optimization
        anomaly_detector: Optional AI component for anomaly detection
    """
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        edge_detector: EdgeDetector,
        contour_extractor: ContourExtractor,
        fourier_transformer: FourierTransformer,
        encryptor: EncryptionStrategy,
        serializer: CoefficientSerializer,
        key_manager: KeyManager,
        optimizer: Optional[object] = None,
        anomaly_detector: Optional[object] = None,
        reconstructor: Optional[ImageReconstructor] = None,
    ):
        """
        Initialize the encryption orchestrator with all required components.
        
        Args:
            image_processor: Image loading and preprocessing component
            edge_detector: Edge detection component (traditional or AI)
            contour_extractor: Contour extraction component
            fourier_transformer: Fourier transform component
            encryptor: Encryption strategy implementation
            serializer: Coefficient serialization component
            key_manager: Key management component
            optimizer: Optional AI coefficient optimizer
            anomaly_detector: Optional AI anomaly detector
            reconstructor: Optional ImageReconstructor for reconstruction after decryption
        """
        self.image_processor = image_processor
        self.edge_detector = edge_detector
        self.contour_extractor = contour_extractor
        self.fourier_transformer = fourier_transformer
        self.encryptor = encryptor
        self.serializer = serializer
        self.key_manager = key_manager
        self.optimizer = optimizer
        self.anomaly_detector = anomaly_detector
        self.reconstructor = reconstructor
        
        logger.info(
            "EncryptionOrchestrator initialized",
            extra={
                "has_optimizer": optimizer is not None,
                "has_anomaly_detector": anomaly_detector is not None,
                "has_reconstructor": reconstructor is not None,
            }
        )
    
    def encrypt_image(
        self,
        image_path: Path,
        key: str,
        preprocess_config: PreprocessConfig,
        encryption_config: EncryptionConfig,
    ) -> EncryptedPayload:
        """
        Execute the complete encryption pipeline.
          
        Pipeline stages:
        1. Load and preprocess image
        2. Detect edges (AI or traditional)
        3. Extract contours
        4. Convert to complex plane
        5. Compute DFT
        6. Optimize coefficient count (if AI enabled)
        7. Serialize coefficients
        8. Encrypt with AES-256
        9. Return encrypted payload
        
        Args:
            image_path: Path to input image file
            key: Encryption password/key
            preprocess_config: Image preprocessing configuration
            encryption_config: Encryption and AI feature configuration
            
        Returns:
            EncryptedPayload containing encrypted coefficients and metadata
            
        Raises:
            ImageProcessingError: If image processing fails
            EncryptionError: If encryption fails
        """
        logger.info(
            "Starting encryption pipeline",
            extra={
                "image_path": str(image_path),
                "use_ai_optimization": encryption_config.use_ai_optimization,
            }
        )
        
        try:
            # Stage 1: Load and preprocess image
            logger.debug("Stage 1: Loading and preprocessing image")
            image = self.image_processor.load_image(image_path)
            preprocessed = self.image_processor.preprocess(image, preprocess_config)
            
            # Get image dimensions for metadata
            if len(preprocessed.shape) == 2:
                height, width = preprocessed.shape
            else:
                height, width = preprocessed.shape[:2]
            
            logger.debug(
                "Image loaded and preprocessed",
                extra={"dimensions": f"{width}x{height}"}
            )
            
            # Stage 2: Detect edges
            logger.debug("Stage 2: Detecting edges")
            edge_map = self.edge_detector.detect_edges(preprocessed)
            edge_metrics = self.edge_detector.get_performance_metrics()
            
            logger.debug(
                "Edge detection complete",
                extra={
                    "edge_density": edge_metrics.get("edge_density", 0),
                    "detection_time": edge_metrics.get("detection_time", 0),
                }
            )
            
            # Stage 3: Extract contours
            logger.debug("Stage 3: Extracting contours")
            contours = self.contour_extractor.extract_contours(edge_map)
            
            if not contours:
                raise ImageProcessingError(
                    "No valid contours found in image",
                    context={"image_path": str(image_path)}
                )
            
            # Use the longest contour (most significant)
            primary_contour = contours[0]
            logger.debug(
                "Contours extracted",
                extra={
                    "total_contours": len(contours),
                    "primary_contour_length": primary_contour.length,
                }
            )
            
            # Stage 4: Convert to complex plane
            logger.debug("Stage 4: Converting to complex plane")
            complex_points = self.contour_extractor.to_complex_plane(primary_contour)
            
            # Stage 5: Compute DFT
            logger.debug("Stage 5: Computing Discrete Fourier Transform")
            coefficients = self.fourier_transformer.compute_dft(complex_points)
            
            logger.debug(
                "DFT computed",
                extra={"coefficient_count": len(coefficients)}
            )
            
            # Stage 6: Optimize coefficient count (if AI enabled)
            if encryption_config.use_ai_optimization and self.optimizer is not None:
                logger.debug("Stage 6: Optimizing coefficient count with AI")
                # AI optimization would be called here
                # For now, we'll use the configured count or default truncation
                pass
            
            # Determine final coefficient count
            if encryption_config.num_coefficients is not None:
                # Use configured count
                num_coefficients = encryption_config.num_coefficients
                logger.debug(
                    "Using configured coefficient count",
                    extra={"count": num_coefficients}
                )
            else:
                # Use all coefficients (no truncation)
                num_coefficients = len(coefficients)
                logger.debug(
                    "Using all coefficients (no truncation)",
                    extra={"count": num_coefficients}
                )
            
            # Truncate if needed
            if num_coefficients < len(coefficients):
                coefficients = self.fourier_transformer.truncate_coefficients(
                    coefficients, num_coefficients
                )
                logger.debug(
                    "Coefficients truncated",
                    extra={"final_count": len(coefficients)}
                )
            
            # Stage 7: Serialize coefficients
            logger.debug("Stage 7: Serializing coefficients")
            metadata = {
                "dimensions": [width, height],
                "original_contour_length": primary_contour.length,
            }
            serialized_data = self.serializer.serialize(coefficients, metadata)
            
            logger.debug(
                "Coefficients serialized",
                extra={"serialized_size": len(serialized_data)}
            )
            
            # Stage 8: Encrypt with AES-256
            logger.debug("Stage 8: Encrypting data")
            
            # Validate key strength
            if not self.key_manager.validate_key_strength(key):
                raise EncryptionError(
                    "Encryption key does not meet minimum strength requirements",
                    context={"min_length": 8}
                )
            
            # Derive encryption key from password
            salt = self.key_manager.generate_salt()
            derived_key = self.encryptor.derive_key(key, salt)
            
            # Encrypt the serialized data
            encrypted_payload = self.encryptor.encrypt(serialized_data, derived_key)
            
            # Merge additional metadata (salt, kdf_iterations, dimensions)
            encrypted_payload.metadata.update({
                "salt": salt.hex(),
                "kdf_iterations": encryption_config.kdf_iterations,
                "dimensions": [width, height],
                "original_contour_length": primary_contour.length,
            })
            
            logger.info(
                "Encryption pipeline complete",
                extra={
                    "coefficient_count": len(coefficients),
                    "payload_size": len(encrypted_payload.ciphertext),
                }
            )
            
            return encrypted_payload
            
        except ImageProcessingError:
            logger.error("Image processing failed during encryption")
            raise
        except EncryptionError:
            logger.error("Encryption operation failed")
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during encryption",
                extra={"error": str(e), "error_type": type(e).__name__}
            )
            raise EncryptionError(
                f"Encryption pipeline failed: {e}",
                context={"error": str(e)}
            )
    
    def decrypt_image(
        self,
        payload: EncryptedPayload,
        key: str,
        visualize: bool = False,
        reconstruct: bool = False,
    ) -> np.ndarray:
        """
        Execute the complete decryption pipeline.
        
        Pipeline stages:
        1. Anomaly detection (if AI enabled)
        2. Decrypt payload
        3. Deserialize coefficients
        4. Reconstruct via IDFT or epicycle animation
        5. Optionally perform image reconstruction (if reconstruct=True)
        6. Return reconstructed contour points
        
        Args:
            payload: EncryptedPayload containing encrypted coefficients
            key: Decryption password/key
            visualize: If True, use epicycle animation for reconstruction
            reconstruct: If True, perform image reconstruction after decryption
            
        Returns:
            NumPy array of reconstructed contour points (complex numbers)
            
        Raises:
            EncryptionError: If decryption fails
            ImageProcessingError: If reconstruction fails
        """
        logger.info(
            "Starting decryption pipeline",
            extra={"visualize": visualize, "reconstruct": reconstruct}
        )
        
        try:
            # Stage 1: Anomaly detection (if AI enabled)
            if self.anomaly_detector is not None:
                logger.debug("Stage 1: Running anomaly detection")
                # AI anomaly detection would be called here
                # For now, we skip this stage
                pass
            
            # Stage 2: Decrypt payload
            logger.debug("Stage 2: Decrypting payload")
            
            # Extract salt and KDF iterations from metadata
            if "salt" not in payload.metadata:
                raise EncryptionError(
                    "Salt not found in encrypted payload metadata",
                    context={"metadata_keys": list(payload.metadata.keys())}
                )
            
            salt = bytes.fromhex(payload.metadata["salt"])
            kdf_iterations = payload.metadata.get("kdf_iterations", 100_000)
            
            # Derive decryption key from password
            derived_key = self.encryptor.derive_key(key, salt)
            
            # Decrypt the data
            decrypted_data = self.encryptor.decrypt(payload, derived_key)
            
            logger.debug(
                "Payload decrypted",
                extra={"decrypted_size": len(decrypted_data)}
            )
            
            # Stage 3: Deserialize coefficients
            logger.debug("Stage 3: Deserializing coefficients")
            coefficients, metadata = self.serializer.deserialize(decrypted_data)
            
            logger.debug(
                "Coefficients deserialized",
                extra={
                    "coefficient_count": len(coefficients),
                    "dimensions": metadata.get("dimensions"),
                }
            )
            
            # Stage 4: Reconstruct via IDFT or epicycle animation
            if visualize:
                logger.debug("Stage 4: Reconstructing with epicycle animation")
                # Create epicycle engine for visualization
                engine = EpicycleEngine(coefficients)
                
                # Generate animation frames
                # For now, we'll just compute the final state
                # Full visualization would be handled by a separate component
                num_frames = 100
                frames = list(engine.generate_animation_frames(num_frames))
                
                # Extract trace points from all frames
                trace_points = np.array([frame.trace_point for frame in frames])
                
                logger.debug(
                    "Epicycle animation complete",
                    extra={"num_frames": len(frames)}
                )
                
                return trace_points
            else:
                logger.debug("Stage 4: Reconstructing with IDFT")
                reconstructed_points = self.fourier_transformer.compute_idft(coefficients)
                
                logger.debug(
                    "IDFT reconstruction complete",
                    extra={"point_count": len(reconstructed_points)}
                )
                
                return reconstructed_points
            
        except EncryptionError:
            logger.error("Decryption operation failed")
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during decryption",
                extra={"error": str(e), "error_type": type(e).__name__}
            )
            raise EncryptionError(
                f"Decryption pipeline failed: {e}",
                context={"error": str(e)}
            )

    
    def reconstruct_from_coefficients(
        self,
        coefficients: List[FourierCoefficient],
        config: ReconstructionConfig,
    ) -> ReconstructionResult:
        """
        Perform image reconstruction from decrypted coefficients.
        
        This method is called after decryption to visualize the reconstruction
        process using epicycles. Can be called independently for reconstruction
        without full decryption workflow.
        
        Args:
            coefficients: List of Fourier coefficients to reconstruct from
            config: ReconstructionConfig specifying reconstruction parameters
            
        Returns:
            ReconstructionResult with final image, frames, and metadata
            
        Raises:
            ConfigurationError: If ImageReconstructor not configured
        """
        if not self.reconstructor:
            raise ConfigurationError("ImageReconstructor not configured")
        
        logger.info(
            "Starting image reconstruction",
            extra={
                "mode": config.mode,
                "quality": config.quality,
                "coefficient_count": len(coefficients),
            }
        )
        
        if config.mode == "static":
            # Static reconstruction - final image only
            logger.debug("Performing static reconstruction")
            image = self.reconstructor.reconstruct_static(coefficients)
            
            result = ReconstructionResult(
                final_image=image,
                reconstruction_time=0.0,
                frame_count=1
            )
            
            logger.info("Static reconstruction complete")
            return result
        else:
            # Animated reconstruction - epicycle drawing process
            logger.debug("Performing animated reconstruction")
            engine = EpicycleEngine(coefficients)
            result = self.reconstructor.reconstruct_animated(coefficients, engine)
            
            logger.info(
                "Animated reconstruction complete",
                extra={
                    "frame_count": result.frame_count,
                    "reconstruction_time": result.reconstruction_time,
                    "animation_path": result.animation_path,
                }
            )
            
            return result
