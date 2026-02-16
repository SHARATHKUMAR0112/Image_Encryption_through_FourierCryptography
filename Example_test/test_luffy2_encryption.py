#!/usr/bin/env python3
"""
Example test: Fourier series sketch of luffy2 image with encryption data logging.

This script demonstrates:
1. Loading the luffy2.png image
2. Processing it through the Fourier encryption pipeline
3. Sketching the image using Fourier series (epicycles)
4. Logging all encryption data to a dedicated log file
"""

import logging
from pathlib import Path
import numpy as np

from fourier_encryption.config.logging_config import setup_logging, get_logger, LOGS_DIR
from fourier_encryption.config.paths import get_test_image
from fourier_encryption.application.orchestrator import EncryptionOrchestrator
from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.config.settings import PreprocessConfig, EncryptionConfig

# Setup logging with file output
setup_logging(level="INFO", enable_file_logging=True)
logger = get_logger(__name__)


def log_encryption_data(encrypted_payload, output_file: Path):
    """Log detailed encryption data to a dedicated file."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FOURIER-BASED IMAGE ENCRYPTION DATA - LUFFY2\n")
        f.write("=" * 80 + "\n\n")
        
        # Ciphertext information
        f.write("CIPHERTEXT DATA:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Size: {len(encrypted_payload.ciphertext)} bytes\n")
        f.write(f"Hex (first 64 bytes): {encrypted_payload.ciphertext[:64].hex()}\n")
        f.write(f"Hex (last 64 bytes): {encrypted_payload.ciphertext[-64:].hex()}\n\n")
        
        # IV (Initialization Vector)
        f.write("INITIALIZATION VECTOR (IV):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Size: {len(encrypted_payload.iv)} bytes\n")
        f.write(f"Hex: {encrypted_payload.iv.hex()}\n\n")
        
        # HMAC
        f.write("HMAC (Message Authentication Code):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Size: {len(encrypted_payload.hmac)} bytes\n")
        f.write(f"Hex: {encrypted_payload.hmac.hex()}\n\n")
        
        # Metadata
        f.write("METADATA:\n")
        f.write("-" * 80 + "\n")
        for key, value in encrypted_payload.metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Statistics
        f.write("ENCRYPTION STATISTICS:\n")
        f.write("-" * 80 + "\n")
        total_size = len(encrypted_payload.ciphertext) + len(encrypted_payload.iv) + len(encrypted_payload.hmac)
        f.write(f"Total encrypted payload size: {total_size} bytes\n")
        f.write(f"Encryption algorithm: AES-256-CBC\n")
        f.write(f"Key derivation: PBKDF2-HMAC-SHA256\n")
        f.write(f"Authentication: HMAC-SHA256\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF ENCRYPTION DATA\n")
        f.write("=" * 80 + "\n")


def main():
    """Run the luffy2 Fourier encryption example."""
    logger.info("=" * 80)
    logger.info("FOURIER SERIES SKETCH - LUFFY2 IMAGE ENCRYPTION TEST")
    logger.info("=" * 80)
    
    # Step 1: Load the luffy2 image
    logger.info("\n[Step 1] Loading luffy2.png image...")
    try:
        image_path = get_test_image("luffy2.png")
        logger.info(f"[OK] Image loaded: {image_path}")
        logger.info(f"  File size: {image_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        logger.error(f"[FAIL] Failed to load image: {e}")
        return 1
    
    # Step 2: Initialize the encryption orchestrator
    logger.info("\n[Step 2] Initializing Fourier encryption system...")
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
    logger.info("[OK] Orchestrator initialized with all components")
    
    # Step 3: Configure preprocessing and encryption
    logger.info("\n[Step 3] Configuring encryption parameters...")
    preprocess_config = PreprocessConfig(
        target_size=(400, 400),
        maintain_aspect_ratio=True,
        normalize=True,
        denoise=True,
    )
    
    encryption_config = EncryptionConfig(
        num_coefficients=100,  # More coefficients for better sketch quality
        use_ai_edge_detection=False,
        use_ai_optimization=False,
        use_anomaly_detection=False,
        kdf_iterations=100_000,
        visualization_enabled=False,
    )
    
    logger.info(f"  Target size: {preprocess_config.target_size}")
    logger.info(f"  Fourier coefficients: {encryption_config.num_coefficients}")
    logger.info(f"  KDF iterations: {encryption_config.kdf_iterations}")
    
    # Step 4: Encrypt the image
    logger.info("\n[Step 4] Encrypting luffy2 image using Fourier series...")
    encryption_key = "SecureLuffy2Password!2026"
    
    try:
        encrypted_payload = orchestrator.encrypt_image(
            image_path,
            encryption_key,
            preprocess_config,
            encryption_config,
        )
        logger.info("[OK] Image encrypted successfully!")
        logger.info(f"  Ciphertext size: {len(encrypted_payload.ciphertext)} bytes")
        logger.info(f"  IV size: {len(encrypted_payload.iv)} bytes")
        logger.info(f"  HMAC size: {len(encrypted_payload.hmac)} bytes")
        logger.info(f"  Metadata keys: {list(encrypted_payload.metadata.keys())}")
    except Exception as e:
        logger.error(f"[FAIL] Encryption failed: {e}", exc_info=True)
        return 1
    
    # Step 5: Log encryption data to dedicated file
    logger.info("\n[Step 5] Writing encryption data to log file...")
    encryption_log_file = LOGS_DIR / "luffy2_encryption_data.log"
    try:
        log_encryption_data(encrypted_payload, encryption_log_file)
        logger.info(f"[OK] Encryption data logged to: {encryption_log_file}")
    except Exception as e:
        logger.error(f"[FAIL] Failed to write encryption log: {e}")
        return 1
    
    # Step 6: Decrypt and verify
    logger.info("\n[Step 6] Decrypting to verify Fourier series reconstruction...")
    try:
        reconstructed_points = orchestrator.decrypt_image(
            encrypted_payload,
            encryption_key,
            visualize=False,
        )
        logger.info("[OK] Image decrypted successfully!")
        logger.info(f"  Reconstructed points: {len(reconstructed_points)}")
        logger.info(f"  Point shape: {reconstructed_points.shape}")
        logger.info(f"  Point dtype: {reconstructed_points.dtype}")
        
        # Log some sample points
        if len(reconstructed_points) > 0:
            logger.info(f"  Sample Fourier coefficients (first 5):")
            for i, coeff in enumerate(reconstructed_points[:5]):
                logger.info(f"    Coefficient {i}: {coeff.real:.2f} + {coeff.imag:.2f}i")
    except Exception as e:
        logger.error(f"[FAIL] Decryption failed: {e}", exc_info=True)
        return 1
    
    # Step 7: Test wrong key rejection
    logger.info("\n[Step 7] Testing security - wrong key rejection...")
    try:
        orchestrator.decrypt_image(
            encrypted_payload,
            "WrongPassword123!",
            visualize=False,
        )
        logger.error("[FAIL] SECURITY FAILURE: Wrong key was accepted!")
        return 1
    except Exception:
        logger.info("[OK] Wrong key correctly rejected - security verified")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY - LUFFY2 FOURIER ENCRYPTION")
    logger.info("=" * 80)
    logger.info("[OK] Image loaded and processed")
    logger.info("[OK] Fourier series coefficients computed")
    logger.info("[OK] Image encrypted with AES-256")
    logger.info("[OK] Encryption data logged to file")
    logger.info("[OK] Image decrypted and reconstructed")
    logger.info("[OK] Security verified (wrong key rejected)")
    logger.info("\nOUTPUT FILES:")
    logger.info(f"  - Main log: {LOGS_DIR / 'latest.log'}")
    logger.info(f"  - Encryption data: {encryption_log_file}")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
