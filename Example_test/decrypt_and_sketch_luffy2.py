#!/usr/bin/env python3
"""
Decrypt luffy2 and create Fourier series sketching visualization.

This script:
1. Decrypts the luffy2 encrypted data
2. Creates a live animation showing epicycles sketching the image
3. Saves frames as images for review
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from fourier_encryption.config.logging_config import setup_logging, get_logger, LOGS_DIR
from fourier_encryption.config.paths import get_test_image, PROJECT_ROOT
from fourier_encryption.application.orchestrator import EncryptionOrchestrator
from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.config.settings import PreprocessConfig, EncryptionConfig

# Setup logging
setup_logging(level="INFO", enable_file_logging=True)
logger = get_logger(__name__)


def create_sketch_frames(coefficients, num_frames=100, output_dir=None):
    """
    Create frames showing the Fourier series sketching process.
    
    Args:
        coefficients: List of FourierCoefficient objects
        num_frames: Number of frames to generate
        output_dir: Directory to save frames (optional)
    
    Returns:
        List of frame images
    """
    logger.info(f"Generating {num_frames} sketch frames with {len(coefficients)} epicycles")
    
    # Create epicycle engine
    engine = EpicycleEngine(coefficients)
    
    # Generate all frames
    frames = list(engine.generate_animation_frames(num_frames))
    
    # Extract trace points
    trace_points = np.array([frame.trace_point for frame in frames])
    x_coords = trace_points.real
    y_coords = trace_points.imag
    
    # Determine plot bounds
    padding = 50
    x_min, x_max = x_coords.min() - padding, x_coords.max() + padding
    y_min, y_max = y_coords.min() - padding, y_coords.max() + padding
    
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate key frames to show progression
    key_frame_indices = [
        0,  # Start
        num_frames // 4,  # 25%
        num_frames // 2,  # 50%
        3 * num_frames // 4,  # 75%
        num_frames - 1  # End
    ]
    
    saved_frames = []
    
    for idx in key_frame_indices:
        frame = frames[idx]
        progress = (idx + 1) / num_frames * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.set_facecolor('#0a0a0a')
        fig.patch.set_facecolor('#0a0a0a')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.invert_yaxis()
        ax.axis('off')
        
        # Plot traced path up to this frame
        traced_x = [frames[i].trace_point.real for i in range(idx + 1)]
        traced_y = [frames[i].trace_point.imag for i in range(idx + 1)]
        ax.plot(traced_x, traced_y, 'cyan', linewidth=2, alpha=0.8)
        
        # Plot epicycles at current frame
        for i, (pos, coeff) in enumerate(zip(frame.positions, coefficients)):
            # Draw circle
            circle = plt.Circle((pos.real, pos.imag), coeff.amplitude,
                              fill=False, color='yellow', linewidth=1, alpha=0.3)
            ax.add_patch(circle)
            
            # Draw line to next epicycle
            if i < len(frame.positions) - 1:
                next_pos = frame.positions[i + 1]
                ax.plot([pos.real, next_pos.real], [pos.imag, next_pos.imag],
                       'yellow', linewidth=1, alpha=0.5)
            else:
                # Last epicycle to trace point
                ax.plot([pos.real, frame.trace_point.real],
                       [pos.imag, frame.trace_point.imag],
                       'red', linewidth=2, alpha=0.7)
        
        # Add title
        ax.text(0.5, 0.98, f'Fourier Series Sketching - Progress: {progress:.1f}%',
               transform=fig.transFigure, ha='center', va='top',
               color='white', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        # Save frame if output directory specified
        if output_dir:
            frame_path = output_dir / f"frame_{idx:04d}.png"
            plt.savefig(frame_path, dpi=100, facecolor='#0a0a0a')
            logger.info(f"  Saved frame {idx + 1}/{num_frames} ({progress:.1f}%) to {frame_path}")
            saved_frames.append(frame_path)
        
        plt.close(fig)
    
    return saved_frames


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("DECRYPT AND SKETCH LUFFY2 - FOURIER SERIES VISUALIZATION")
    logger.info("=" * 80)
    
    # Step 1: Load image
    logger.info("\n[Step 1] Loading luffy2.png...")
    try:
        image_path = get_test_image("luffy2.png")
        logger.info(f"[OK] Image loaded: {image_path}")
    except Exception as e:
        logger.error(f"[FAIL] Failed to load image: {e}")
        return 1
    
    # Step 2: Initialize system
    logger.info("\n[Step 2] Initializing encryption system...")
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
    logger.info("[OK] System initialized")
    
    # Step 3: Encrypt
    logger.info("\n[Step 3] Encrypting image...")
    preprocess_config = PreprocessConfig(
        target_size=(400, 400),
        maintain_aspect_ratio=True,
        normalize=True,
        denoise=True,
    )
    
    encryption_config = EncryptionConfig(
        num_coefficients=100,
        use_ai_edge_detection=False,
        use_ai_optimization=False,
        use_anomaly_detection=False,
        kdf_iterations=100_000,
        visualization_enabled=False,
    )
    
    encryption_key = "SecureLuffy2Password!2026"
    
    try:
        encrypted_payload = orchestrator.encrypt_image(
            image_path,
            encryption_key,
            preprocess_config,
            encryption_config,
        )
        logger.info("[OK] Image encrypted")
        logger.info(f"  Payload size: {len(encrypted_payload.ciphertext)} bytes")
    except Exception as e:
        logger.error(f"[FAIL] Encryption failed: {e}", exc_info=True)
        return 1
    
    # Step 4: Decrypt
    logger.info("\n[Step 4] Decrypting to extract Fourier coefficients...")
    try:
        salt = bytes.fromhex(encrypted_payload.metadata["salt"])
        derived_key = orchestrator.encryptor.derive_key(encryption_key, salt)
        decrypted_data = orchestrator.encryptor.decrypt(encrypted_payload, derived_key)
        coefficients, metadata = orchestrator.serializer.deserialize(decrypted_data)
        
        logger.info("[OK] Decryption successful!")
        logger.info(f"  Recovered {len(coefficients)} Fourier coefficients")
        logger.info(f"  Dimensions: {metadata.get('dimensions')}")
    except Exception as e:
        logger.error(f"[FAIL] Decryption failed: {e}", exc_info=True)
        return 1
    
    # Step 5: Create visualization
    logger.info("\n[Step 5] Creating Fourier series sketch visualization...")
    output_dir = PROJECT_ROOT / "output" / "luffy2_sketch"
    
    try:
        saved_frames = create_sketch_frames(
            coefficients,
            num_frames=200,
            output_dir=output_dir
        )
        logger.info(f"[OK] Created {len(saved_frames)} key frames")
        logger.info(f"  Output directory: {output_dir}")
    except Exception as e:
        logger.error(f"[FAIL] Visualization failed: {e}", exc_info=True)
        return 1
    
    # Step 6: Create comparison image
    logger.info("\n[Step 6] Creating before/after comparison...")
    try:
        # Load original image
        original = cv2.imread(str(image_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Load final sketch frame
        final_frame = cv2.imread(str(saved_frames[-1]))
        final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        
        # Create comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('#0a0a0a')
        
        ax1.imshow(original)
        ax1.set_title('Original Luffy2 Image', color='white', fontsize=14)
        ax1.axis('off')
        
        ax2.imshow(final_frame)
        ax2.set_title('Fourier Series Reconstruction', color='white', fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        comparison_path = output_dir / "comparison.png"
        plt.savefig(comparison_path, dpi=150, facecolor='#0a0a0a')
        plt.close()
        
        logger.info(f"[OK] Comparison saved to: {comparison_path}")
    except Exception as e:
        logger.warning(f"Could not create comparison: {e}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput files saved to: {output_dir}")
    logger.info(f"  - {len(saved_frames)} key frames showing sketch progression")
    logger.info(f"  - comparison.png showing original vs reconstruction")
    logger.info("\nThe Fourier series successfully reconstructed the image!")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
