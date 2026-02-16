#!/usr/bin/env python3
"""
Example script demonstrating how to use test images from the images/ folder.

This script shows:
1. How to access test images from the images/ folder
2. How logging is automatically saved to timestamped files in logs/ folder
3. Basic encryption/decryption workflow with real images
"""

import logging
from pathlib import Path

from fourier_encryption.config.paths import (
    get_test_image,
    list_test_images,
    AVAILABLE_IMAGES,
    LOGS_DIR,
)
from fourier_encryption.config.logging_config import setup_logging
from fourier_encryption.application.orchestrator import EncryptionOrchestrator
from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.config.settings import PreprocessConfig, EncryptionConfig

# Setup logging (automatically logs to timestamped file in logs/ folder)
setup_logging(level="INFO", enable_file_logging=True)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating image usage and logging."""
    logger.info("=" * 60)
    logger.info("Fourier Image Encryption - Test Images Demo")
    logger.info("=" * 60)
    
    # List all available test images
    logger.info("\nAvailable test images:")
    test_images = list_test_images()
    for i, img_path in enumerate(test_images, 1):
        logger.info(f"  {i}. {img_path.name} ({img_path.stat().st_size / 1024:.1f} KB)")
    
    # Show predefined image aliases
    logger.info("\nPredefined image aliases:")
    for alias, filename in AVAILABLE_IMAGES.items():
        logger.info(f"  {alias}: {filename}")
    
    # Use a specific test image
    try:
        test_image = get_test_image("luffy_1.png")
        logger.info(f"\nUsing test image: {test_image}")
        
        # Create orchestrator
        logger.info("\nInitializing encryption orchestrator...")
        orchestrator = EncryptionOrchestrator(
            image_processor=OpenCVImageProcessor(),
            edge_detector=CannyEdgeDetector(),
            contour_extractor=ContourExtractor(),
            fourier_transformer=FourierTransformer(),
            encryptor=AES256Encryptor(),
            serializer=CoefficientSerializer(),
            key_manager=KeyManager(),
            optimizer=None,
            anomaly_detector=None,
        )
        
        # Configure encryption
        preprocess_config = PreprocessConfig(
            target_size=(400, 400),
            maintain_aspect_ratio=True,
            normalize=False,
            denoise=False,
        )
        
        encryption_config = EncryptionConfig(
            num_coefficients=100,
            use_ai_edge_detection=False,
            use_ai_optimization=False,
            use_anomaly_detection=False,
            kdf_iterations=100_000,
            visualization_enabled=False,
        )
        
        # Encrypt the image
        logger.info("\nEncrypting image...")
        key = "SecureTestPassword123!"
        encrypted_payload = orchestrator.encrypt_image(
            test_image,
            key,
            preprocess_config,
            encryption_config,
        )
        
        logger.info(f"Encryption successful!")
        logger.info(f"  Ciphertext size: {len(encrypted_payload.ciphertext)} bytes")
        logger.info(f"  IV: {encrypted_payload.iv.hex()[:32]}...")
        logger.info(f"  Dimensions: {encrypted_payload.metadata['dimensions']}")
        
        # Decrypt the image
        logger.info("\nDecrypting image...")
        reconstructed_points = orchestrator.decrypt_image(
            encrypted_payload,
            key,
            visualize=False,
        )
        
        logger.info(f"Decryption successful!")
        logger.info(f"  Reconstructed {len(reconstructed_points)} points")
        
        logger.info("\n" + "=" * 60)
        logger.info("Demo completed successfully!")
        logger.info(f"Check logs folder for detailed logs: {LOGS_DIR}")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"Image not found: {e}")
        logger.info("\nAvailable images:")
        for img in test_images:
            logger.info(f"  - {img.name}")
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)


if __name__ == "__main__":
    main()
