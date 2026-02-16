"""
Performance Benchmark Script for Fourier-Based Image Encryption System

This script benchmarks key performance metrics:
- Image processing time vs. resolution
- DFT computation time vs. number of points
- Encryption time vs. coefficient count
- Memory usage profiling

Requirements from AC 3.13.1, 3.13.2, 3.3.5, 3.4.7, 3.1.5:
- 1080p images process end-to-end within 15 seconds
- Memory usage stays under 2GB
- Animation maintains 30+ FPS
- Encryption completes within 2 seconds for 500 coefficients
- Edge extraction handles 4K images within 5 seconds
"""

import time
import numpy as np
import psutil
import os
from pathlib import Path

# Import system components
from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.core.edge_detector import CannyEdgeDetector
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.encryption.aes_encryptor import AES256Encryptor
from fourier_encryption.encryption.key_manager import KeyManager
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.config.settings import PreprocessConfig


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_image_processing():
    """Benchmark image processing time vs. resolution."""
    print("\n=== Image Processing Benchmark ===")
    
    resolutions = [
        (640, 480, "VGA"),
        (1280, 720, "720p"),
        (1920, 1080, "1080p"),
        (3840, 2160, "4K")
    ]
    
    processor = OpenCVImageProcessor()
    edge_detector = CannyEdgeDetector()
    config = PreprocessConfig(target_size=(1920, 1080), maintain_aspect_ratio=False, normalize=False, denoise=False)
    
    for width, height, name in resolutions:
        # Create synthetic test image
        test_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        
        start_time = time.time()
        start_mem = get_memory_usage_mb()
        
        # Process image
        preprocessed = processor.preprocess(test_image, config)
        edges = edge_detector.detect_edges(preprocessed)
        
        end_time = time.time()
        end_mem = get_memory_usage_mb()
        
        elapsed = end_time - start_time
        mem_delta = end_mem - start_mem
        
        print(f"{name} ({width}x{height}): {elapsed:.3f}s, Memory: +{mem_delta:.2f}MB")
        
        # Check requirements
        if name == "1080p" and elapsed > 15:
            print(f"  ⚠️  WARNING: 1080p processing exceeded 15s requirement")
        if name == "4K" and elapsed > 5:
            print(f"  ⚠️  WARNING: 4K edge extraction exceeded 5s requirement")


def benchmark_dft_computation():
    """Benchmark DFT computation time vs. number of points."""
    print("\n=== DFT Computation Benchmark ===")
    
    point_counts = [10, 50, 100, 500, 1000, 5000]
    transformer = FourierTransformer()
    
    for num_points in point_counts:
        # Generate random contour points
        points = np.random.rand(num_points) + 1j * np.random.rand(num_points)
        
        start_time = time.time()
        coefficients = transformer.compute_dft(points)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"{num_points} points: {elapsed:.4f}s ({elapsed*1000:.2f}ms)")


def benchmark_encryption():
    """Benchmark encryption time vs. coefficient count."""
    print("\n=== Encryption Benchmark ===")
    
    coefficient_counts = [10, 50, 100, 500, 1000]
    encryptor = AES256Encryptor(kdf_iterations=100_000)
    serializer = CoefficientSerializer()
    key_manager = KeyManager()
    
    # Generate encryption key
    password = "test_password_123"
    salt = key_manager.generate_salt()
    key = encryptor.derive_key(password, salt)
    
    for count in coefficient_counts:
        # Generate random coefficients with consistent amplitude/phase/complex_value
        from fourier_encryption.models.data_models import FourierCoefficient
        coefficients = []
        for i in range(count):
            amplitude = float(np.random.rand())
            phase = float(np.random.rand() * 2 * np.pi - np.pi)
            # Calculate complex value from amplitude and phase
            complex_value = amplitude * np.exp(1j * phase)
            
            coefficients.append(FourierCoefficient(
                frequency=i,
                amplitude=amplitude,
                phase=phase,
                complex_value=complex_value
            ))
        
        # Serialize
        metadata = {"version": "1.0", "count": count, "dimensions": [100, 100]}
        serialized = serializer.serialize(coefficients, metadata)
        
        # Encrypt
        start_time = time.time()
        encrypted = encryptor.encrypt(serialized, key)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"{count} coefficients: {elapsed:.4f}s ({elapsed*1000:.2f}ms)")
        
        # Check requirement
        if count == 500 and elapsed > 2:
            print(f"  ⚠️  WARNING: 500 coefficient encryption exceeded 2s requirement")


def benchmark_memory_usage():
    """Profile memory usage during typical operations."""
    print("\n=== Memory Usage Benchmark ===")
    
    initial_mem = get_memory_usage_mb()
    print(f"Initial memory: {initial_mem:.2f}MB")
    
    # Create large test image (1080p)
    test_image = np.random.randint(0, 256, (1080, 1920), dtype=np.uint8)
    mem_after_image = get_memory_usage_mb()
    print(f"After 1080p image creation: {mem_after_image:.2f}MB (+{mem_after_image - initial_mem:.2f}MB)")
    
    # Process image
    processor = OpenCVImageProcessor()
    edge_detector = CannyEdgeDetector()
    config = PreprocessConfig(target_size=(1920, 1080), maintain_aspect_ratio=False, normalize=False, denoise=False)
    preprocessed = processor.preprocess(test_image, config)
    edges = edge_detector.detect_edges(preprocessed)
    mem_after_processing = get_memory_usage_mb()
    print(f"After image processing: {mem_after_processing:.2f}MB (+{mem_after_processing - initial_mem:.2f}MB)")
    
    # Extract contours and compute DFT
    extractor = ContourExtractor()
    contours = extractor.extract_contours(edges)
    if contours:
        complex_points = extractor.to_complex_plane(contours[0])
        transformer = FourierTransformer()
        coefficients = transformer.compute_dft(complex_points)
        mem_after_dft = get_memory_usage_mb()
        print(f"After DFT computation: {mem_after_dft:.2f}MB (+{mem_after_dft - initial_mem:.2f}MB)")
    
    peak_mem = get_memory_usage_mb()
    print(f"\nPeak memory usage: {peak_mem:.2f}MB")
    
    # Check requirement
    if peak_mem > 2048:
        print(f"  ⚠️  WARNING: Memory usage exceeded 2GB requirement")
    else:
        print(f"  ✓ Memory usage within 2GB requirement")


def main():
    """Run all performance benchmarks."""
    print("=" * 60)
    print("Fourier-Based Image Encryption System")
    print("Performance Benchmark Suite")
    print("=" * 60)
    
    try:
        benchmark_image_processing()
        benchmark_dft_computation()
        benchmark_encryption()
        benchmark_memory_usage()
        
        print("\n" + "=" * 60)
        print("Benchmark Complete")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
