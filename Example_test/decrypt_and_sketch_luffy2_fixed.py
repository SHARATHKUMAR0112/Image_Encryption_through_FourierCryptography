#!/usr/bin/env python3
"""
Decrypt luffy2 and create CORRECT Fourier series sketching visualization.

This script properly reconstructs the image contour using IDFT.
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
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.config.settings import PreprocessConfig, EncryptionConfig

# Setup logging
setup_logging(level="INFO", enable_file_logging=True)
logger = get_logger(__name__)


def create_progressive_sketch(coefficients, fourier_transformer, num_steps=10, output_dir=None):
    """
    Create frames showing progressive reconstruction using IDFT.
    
    Args:
        coefficients: List of FourierCoefficient objects
        fourier_transformer: FourierTransformer instance
        num_steps: Number of progressive steps to show
        output_dir: Directory to save frames
    
    Returns:
        List of saved frame paths
    """
    logger.info(f"Creating progressive sketch with {len(coefficients)} coefficients")
    
    # Sort coefficients by amplitude (most significant first)
    sorted_coeffs = fourier_transformer.sort_by_amplitude(coefficients)
    
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    saved_frames = []
    
    # Create progressive reconstructions
    step_sizes = np.linspace(10, len(sorted_coeffs), num_steps, dtype=int)
    
    for step_idx, num_coeffs in enumerate(step_sizes):
        # Take top N coefficients
        truncated_coeffs = sorted_coeffs[:num_coeffs]
        
        # Reconstruct using IDFT
        reconstructed_points = fourier_transformer.compute_idft(truncated_coeffs)
        
        # Extract x and y coordinates
        x_coords = reconstructed_points.real
        y_coords = reconstructed_points.imag
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        ax.set_facecolor('#0a0a0a')
        fig.patch.set_facecolor('#0a0a0a')
        
        # Plot the reconstructed contour
        ax.plot(x_coords, y_coords, 'cyan', linewidth=2, alpha=0.9)
        ax.scatter(x_coords[0], y_coords[0], c='red', s=100, zorder=5, 
                  label='Start Point', alpha=0.8)
        
        # Set limits with padding
        padding = 50
        x_min, x_max = x_coords.min() - padding, x_coords.max() + padding
        y_min, y_max = y_coords.min() - padding, y_coords.max() + padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        ax.axis('off')
        
        # Add title
        progress = (step_idx + 1) / num_steps * 100
        ax.text(0.5, 0.98, 
               f'Fourier Reconstruction - {num_coeffs}/{len(coefficients)} Coefficients ({progress:.0f}%)',
               transform=fig.transFigure, ha='center', va='top',
               color='white', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        # Save frame
        if output_dir:
            frame_path = output_dir / f"reconstruction_{step_idx:02d}_{num_coeffs:04d}coeffs.png"
            plt.savefig(frame_path, dpi=150, facecolor='#0a0a0a')
            logger.info(f"  Saved frame {step_idx + 1}/{num_steps}: {num_coeffs} coefficients")
            saved_frames.append(frame_path)
        
        plt.close(fig)
    
    return saved_frames


def create_full_reconstruction_comparison(coefficients, fourier_transformer, 
                                         original_image_path, output_dir):
    """
    Create a comparison showing original image and full reconstruction.
    """
    logger.info("Creating full reconstruction comparison...")
    
    # Full reconstruction using all coefficients
    reconstructed_points = fourier_transformer.compute_idft(coefficients)
    
    # Extract coordinates
    x_coords = reconstructed_points.real
    y_coords = reconstructed_points.imag
    
    # Load original image
    original = cv2.imread(str(original_image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Original image
    ax1.imshow(original)
    ax1.set_title('Original Luffy2 Image', color='white', fontsize=16, pad=20)
    ax1.axis('off')
    
    # Reconstructed contour
    ax2.set_aspect('equal')
    ax2.set_facecolor('#0a0a0a')
    ax2.plot(x_coords, y_coords, 'cyan', linewidth=2, alpha=0.9)
    ax2.scatter(x_coords[0], y_coords[0], c='red', s=100, zorder=5, alpha=0.8)
    
    # Set limits
    padding = 50
    x_min, x_max = x_coords.min() - padding, x_coords.max() + padding
    y_min, y_max = y_coords.min() - padding, y_coords.max() + padding
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.invert_yaxis()
    ax2.axis('off')
    ax2.set_title(f'Fourier Reconstruction ({len(coefficients)} coefficients)', 
                 color='white', fontsize=16, pad=20)
    
    plt.tight_layout()
    
    # Save
    comparison_path = output_dir / "full_comparison.png"
    plt.savefig(comparison_path, dpi=150, facecolor='#0a0a0a')
    plt.close()
    
    logger.info(f"  Saved comparison to: {comparison_path}")
    return comparison_path


def create_overlay_comparison(coefficients, fourier_transformer, 
                              original_image_path, output_dir):
    """
    Create an overlay showing the reconstruction on top of the original image.
    """
    logger.info("Creating overlay comparison...")
    
    # Full reconstruction
    reconstructed_points = fourier_transformer.compute_idft(coefficients)
    x_coords = reconstructed_points.real
    y_coords = reconstructed_points.imag
    
    # Load original image
    original = cv2.imread(str(original_image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Show original image
    ax.imshow(original, alpha=0.6)
    
    # Overlay reconstruction
    ax.plot(x_coords, y_coords, 'cyan', linewidth=3, alpha=0.9, label='Fourier Reconstruction')
    ax.scatter(x_coords[0], y_coords[0], c='red', s=150, zorder=5, alpha=0.9, label='Start Point')
    
    ax.set_title(f'Overlay: Original Image + Fourier Reconstruction', 
                color='white', fontsize=16, pad=20)
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    overlay_path = output_dir / "overlay_comparison.png"
    plt.savefig(overlay_path, dpi=150, facecolor='white')
    plt.close()
    
    logger.info(f"  Saved overlay to: {overlay_path}")
    return overlay_path


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("DECRYPT AND SKETCH LUFFY2 - CORRECTED FOURIER RECONSTRUCTION")
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
    
    # Step 3: Encrypt with MORE coefficients for better quality
    logger.info("\n[Step 3] Encrypting image...")
    preprocess_config = PreprocessConfig(
        target_size=(400, 400),
        maintain_aspect_ratio=True,
        normalize=True,
        denoise=True,
    )
    
    encryption_config = EncryptionConfig(
        num_coefficients=200,  # More coefficients for better reconstruction
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
    
    # Step 5: Create progressive reconstruction
    logger.info("\n[Step 5] Creating progressive reconstruction frames...")
    output_dir = PROJECT_ROOT / "output" / "luffy2_corrected"
    
    try:
        saved_frames = create_progressive_sketch(
            coefficients,
            orchestrator.fourier_transformer,
            num_steps=10,
            output_dir=output_dir
        )
        logger.info(f"[OK] Created {len(saved_frames)} progressive frames")
    except Exception as e:
        logger.error(f"[FAIL] Progressive reconstruction failed: {e}", exc_info=True)
        return 1
    
    # Step 6: Create full comparison
    logger.info("\n[Step 6] Creating full reconstruction comparison...")
    try:
        comparison_path = create_full_reconstruction_comparison(
            coefficients,
            orchestrator.fourier_transformer,
            image_path,
            output_dir
        )
        logger.info("[OK] Comparison created")
    except Exception as e:
        logger.error(f"[FAIL] Comparison failed: {e}", exc_info=True)
        return 1
    
    # Step 7: Create overlay
    logger.info("\n[Step 7] Creating overlay comparison...")
    try:
        overlay_path = create_overlay_comparison(
            coefficients,
            orchestrator.fourier_transformer,
            image_path,
            output_dir
        )
        logger.info("[OK] Overlay created")
    except Exception as e:
        logger.error(f"[FAIL] Overlay failed: {e}", exc_info=True)
        return 1
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CORRECTED VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput files saved to: {output_dir}")
    logger.info(f"  - {len(saved_frames)} progressive reconstruction frames")
    logger.info(f"  - full_comparison.png (side-by-side comparison)")
    logger.info(f"  - overlay_comparison.png (reconstruction overlaid on original)")
    logger.info("\nThe Fourier series correctly reconstructed the image contour!")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
