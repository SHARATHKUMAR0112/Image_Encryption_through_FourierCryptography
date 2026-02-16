#!/usr/bin/env python3
"""
Decrypt luffy2 encrypted data and show live Fourier series sketching animation.

This script:
1. Loads the encrypted payload from the previous test
2. Decrypts the Fourier coefficients
3. Animates the epicycles drawing the image in real-time
4. Shows rotating circles (epicycles) that combine to sketch the image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from fourier_encryption.config.logging_config import setup_logging, get_logger
from fourier_encryption.config.paths import get_test_image
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


def create_live_animation(coefficients, num_frames=200, show_epicycles=True):
    """
    Create a live animation showing Fourier series sketching.
    
    Args:
        coefficients: List of FourierCoefficient objects
        num_frames: Number of frames in the animation
        show_epicycles: Whether to show the rotating epicycles
    """
    logger.info(f"Creating animation with {len(coefficients)} epicycles")
    
    # Create epicycle engine
    engine = EpicycleEngine(coefficients)
    
    # Pre-compute all frames for smooth animation
    logger.info(f"Pre-computing {num_frames} animation frames...")
    frames = list(engine.generate_animation_frames(num_frames))
    
    # Extract all trace points to determine plot bounds
    trace_points = np.array([frame.trace_point for frame in frames])
    x_coords = trace_points.real
    y_coords = trace_points.imag
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    
    # Set plot limits with some padding
    padding = 50
    x_min, x_max = x_coords.min() - padding, x_coords.max() + padding
    y_min, y_max = y_coords.min() - padding, y_coords.max() + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Invert y-axis to match image coordinates
    ax.invert_yaxis()
    
    # Remove axes for cleaner look
    ax.axis('off')
    
    # Initialize plot elements
    trace_line, = ax.plot([], [], 'cyan', linewidth=2, alpha=0.8, label='Traced Path')
    
    if show_epicycles:
        epicycle_circles = []
        epicycle_lines = []
        
        # Create circle and line objects for each epicycle
        for i in range(len(coefficients)):
            circle = plt.Circle((0, 0), 1, fill=False, color='yellow', 
                              linewidth=1, alpha=0.3)
            ax.add_patch(circle)
            epicycle_circles.append(circle)
            
            line, = ax.plot([], [], 'yellow', linewidth=1, alpha=0.5)
            epicycle_lines.append(line)
        
        # Line from last epicycle to trace point
        final_line, = ax.plot([], [], 'red', linewidth=2, alpha=0.7)
    
    # Title with frame counter
    title = ax.text(0.5, 0.98, '', transform=fig.transFigure,
                   ha='center', va='top', color='white', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Store traced points
    traced_x = []
    traced_y = []
    
    def init():
        """Initialize animation."""
        trace_line.set_data([], [])
        if show_epicycles:
            for circle in epicycle_circles:
                circle.set_visible(False)
            for line in epicycle_lines:
                line.set_data([], [])
            final_line.set_data([], [])
        title.set_text('Fourier Series Sketching - Luffy2')
        return [trace_line, title]
    
    def animate(frame_idx):
        """Update animation for each frame."""
        frame = frames[frame_idx]
        
        # Add current trace point to the path
        traced_x.append(frame.trace_point.real)
        traced_y.append(frame.trace_point.imag)
        
        # Update traced path
        trace_line.set_data(traced_x, traced_y)
        
        # Update epicycles if enabled
        if show_epicycles:
            for i, (pos, coeff) in enumerate(zip(frame.positions, coefficients)):
                # Update circle position and radius
                epicycle_circles[i].center = (pos.real, pos.imag)
                epicycle_circles[i].radius = coeff.amplitude
                epicycle_circles[i].set_visible(True)
                
                # Update line from previous position to current
                if i == 0:
                    # First epicycle starts at origin
                    epicycle_lines[i].set_data([0, pos.real], [0, pos.imag])
                else:
                    prev_pos = frame.positions[i-1]
                    epicycle_lines[i].set_data([prev_pos.real, pos.real], 
                                              [prev_pos.imag, pos.imag])
            
            # Update final line from last epicycle to trace point
            if frame.positions:
                last_pos = frame.positions[-1]
                final_line.set_data([last_pos.real, frame.trace_point.real],
                                   [last_pos.imag, frame.trace_point.imag])
        
        # Update title with progress
        progress = (frame_idx + 1) / num_frames * 100
        title.set_text(f'Fourier Series Sketching - Luffy2 | '
                      f'Frame {frame_idx + 1}/{num_frames} ({progress:.1f}%)')
        
        return [trace_line, title]
    
    # Create animation
    logger.info("Starting animation...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=num_frames, interval=50,  # 50ms = 20 FPS
        blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    
    return anim


def main():
    """Main function to decrypt and animate luffy2."""
    logger.info("=" * 80)
    logger.info("DECRYPT AND ANIMATE LUFFY2 - LIVE FOURIER SKETCHING")
    logger.info("=" * 80)
    
    # Step 1: Load the original image and encrypt it
    logger.info("\n[Step 1] Loading and encrypting luffy2.png...")
    try:
        image_path = get_test_image("luffy2.png")
        logger.info(f"[OK] Image loaded: {image_path}")
    except Exception as e:
        logger.error(f"[FAIL] Failed to load image: {e}")
        return 1
    
    # Initialize orchestrator
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
    logger.info("[OK] Orchestrator initialized")
    
    # Configure encryption
    preprocess_config = PreprocessConfig(
        target_size=(400, 400),
        maintain_aspect_ratio=True,
        normalize=True,
        denoise=True,
    )
    
    encryption_config = EncryptionConfig(
        num_coefficients=150,  # More coefficients for better detail
        use_ai_edge_detection=False,
        use_ai_optimization=False,
        use_anomaly_detection=False,
        kdf_iterations=100_000,
        visualization_enabled=False,
    )
    
    logger.info(f"  Fourier coefficients: {encryption_config.num_coefficients}")
    
    # Encrypt the image
    logger.info("\n[Step 3] Encrypting image...")
    encryption_key = "SecureLuffy2Password!2026"
    
    try:
        encrypted_payload = orchestrator.encrypt_image(
            image_path,
            encryption_key,
            preprocess_config,
            encryption_config,
        )
        logger.info("[OK] Image encrypted successfully!")
        logger.info(f"  Encrypted payload size: {len(encrypted_payload.ciphertext)} bytes")
    except Exception as e:
        logger.error(f"[FAIL] Encryption failed: {e}", exc_info=True)
        return 1
    
    # Decrypt and get coefficients
    logger.info("\n[Step 4] Decrypting to extract Fourier coefficients...")
    try:
        # Extract salt and derive key
        salt = bytes.fromhex(encrypted_payload.metadata["salt"])
        derived_key = orchestrator.encryptor.derive_key(encryption_key, salt)
        
        # Decrypt the data
        decrypted_data = orchestrator.encryptor.decrypt(encrypted_payload, derived_key)
        
        # Deserialize coefficients
        coefficients, metadata = orchestrator.serializer.deserialize(decrypted_data)
        
        logger.info("[OK] Decryption successful!")
        logger.info(f"  Recovered {len(coefficients)} Fourier coefficients")
        logger.info("Coefficients" , coefficients)
        logger.info(f"  Image dimensions: {metadata.get('dimensions')}")
        
        # Log some sample coefficients
        logger.info("\n  Sample coefficients:")
        for i, coeff in enumerate(coefficients[:5]):
            logger.info(f"    Coeff {i}: freq={coeff.frequency}, "
                       f"amp={coeff.amplitude:.2f}, phase={coeff.phase:.2f}")
        
    except Exception as e:
        logger.error(f"[FAIL] Decryption failed: {e}", exc_info=True)
        return 1
    
    # Create live animation
    logger.info("\n[Step 5] Creating live Fourier series animation...")
    logger.info("  This will show rotating epicycles sketching the image")
    logger.info("  Close the window to exit")
    
    try:
        create_live_animation(
            coefficients,
            num_frames=300,  # 300 frames for smooth animation
            show_epicycles=True  # Show the rotating circles
        )
        logger.info("[OK] Animation completed!")
    except Exception as e:
        logger.error(f"[FAIL] Animation failed: {e}", exc_info=True)
        return 1
    
    logger.info("\n" + "=" * 80)
    logger.info("ANIMATION COMPLETE")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
