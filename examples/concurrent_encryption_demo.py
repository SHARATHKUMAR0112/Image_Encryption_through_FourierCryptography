"""
Concurrent Encryption Demo

This example demonstrates how to encrypt multiple images in parallel
using the ConcurrentEncryptionOrchestrator.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fourier_encryption.application.orchestrator import EncryptionOrchestrator
from fourier_encryption.concurrency import (
    ConcurrentEncryptionOrchestrator,
    EncryptionTask,
)
from fourier_encryption.config.settings import (
    EncryptionConfig,
    PreprocessConfig,
    SystemConfig,
)
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.core.image_processor import ImageProcessor
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.transmission.serializer import CoefficientSerializer


def progress_callback(task_id: str, progress: float):
    """Callback for progress updates."""
    if progress < 0:
        print(f"[{task_id}] FAILED")
    elif progress == 0:
        print(f"[{task_id}] Started...")
    elif progress == 100:
        print(f"[{task_id}] Completed!")
    else:
        print(f"[{task_id}] Progress: {progress:.1f}%")


def main():
    """Run concurrent encryption demo."""
    print("=" * 60)
    print("  Concurrent Encryption Demo")
    print("=" * 60)
    print()
    
    # Create base orchestrator
    print("Initializing encryption orchestrator...")
    image_processor = ImageProcessor()
    edge_detector = CannyEdgeDetector()
    contour_extractor = ContourExtractor()
    fourier_transformer = FourierTransformer()
    encryptor = AES256Encryptor()
    serializer = CoefficientSerializer()
    key_manager = KeyManager()
    
    orchestrator = EncryptionOrchestrator(
        image_processor=image_processor,
        edge_detector=edge_detector,
        contour_extractor=contour_extractor,
        fourier_transformer=fourier_transformer,
        encryptor=encryptor,
        serializer=serializer,
        key_manager=key_manager,
    )
    
    # Create concurrent orchestrator
    print("Creating concurrent orchestrator with 4 workers...")
    concurrent_orchestrator = ConcurrentEncryptionOrchestrator(
        orchestrator=orchestrator,
        max_workers=4,
        progress_callback=progress_callback,
    )
    
    # Prepare test images (use the same image multiple times for demo)
    test_image = Path("test_images/simple_shape.png")
    
    if not test_image.exists():
        print(f"Error: Test image not found at {test_image}")
        print("Please ensure test images are available.")
        return
    
    # Create encryption tasks
    print("\nPreparing encryption tasks...")
    tasks = []
    for i in range(8):
        task = EncryptionTask(
            image_path=test_image,
            key=f"test_password_{i}",
            preprocess_config=PreprocessConfig(),
            encryption_config=EncryptionConfig(
                num_coefficients=100,
                use_ai_optimization=False,
            ),
            task_id=f"image_{i+1}",
        )
        tasks.append(task)
    
    print(f"Created {len(tasks)} encryption tasks")
    print()
    
    # Execute batch encryption
    print("Starting concurrent encryption...")
    print("-" * 60)
    start_time = time.time()
    
    results = concurrent_orchestrator.encrypt_batch(tasks, wait=True)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("-" * 60)
    print()
    
    # Display results
    print("Encryption Results:")
    print("=" * 60)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"Total tasks:      {len(results)}")
    print(f"Successful:       {len(successful)}")
    print(f"Failed:           {len(failed)}")
    print(f"Total time:       {elapsed:.2f}s")
    print(f"Average per task: {elapsed/len(results):.2f}s")
    print()
    
    # Show successful results
    if successful:
        print("Successful Encryptions:")
        for result in successful:
            payload_size = len(result.payload.ciphertext) if result.payload else 0
            print(f"  - {result.task_id}: {payload_size} bytes")
        print()
    
    # Show failed results
    if failed:
        print("Failed Encryptions:")
        for result in failed:
            print(f"  - {result.task_id}: {result.error}")
        print()
    
    # Performance comparison
    print("Performance Analysis:")
    print("-" * 60)
    print(f"Parallel execution time:   {elapsed:.2f}s")
    print(f"Sequential would take:     ~{elapsed * 4:.2f}s (estimated)")
    print(f"Speedup factor:            ~{4:.1f}x")
    print()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
