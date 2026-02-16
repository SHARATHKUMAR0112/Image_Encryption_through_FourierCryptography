"""
Performance benchmark script for optimized components.

This script benchmarks the critical paths that have been optimized:
- DFT computation
- Image preprocessing
- Serialization/deserialization
"""

import time
import numpy as np
from pathlib import Path

from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.core.image_processor import OpenCVImageProcessor
from fourier_encryption.transmission.serializer import CoefficientSerializer
from fourier_encryption.config.settings import PreprocessConfig
from fourier_encryption.models.data_models import FourierCoefficient


def benchmark_dft_computation():
    """Benchmark DFT computation with various input sizes."""
    print("\n=== DFT Computation Benchmark ===")
    
    transformer = FourierTransformer()
    sizes = [100, 500, 1000, 2000, 5000]
    
    for size in sizes:
        # Generate random complex points
        points = np.random.random(size) + 1j * np.random.random(size)
        
        # Benchmark DFT
        start = time.perf_counter()
        coefficients = transformer.compute_dft(points)
        dft_time = time.perf_counter() - start
        
        # Benchmark sorting
        start = time.perf_counter()
        sorted_coeffs = transformer.sort_by_amplitude(coefficients)
        sort_time = time.perf_counter() - start
        
        # Benchmark IDFT
        start = time.perf_counter()
        reconstructed = transformer.compute_idft(coefficients)
        idft_time = time.perf_counter() - start
        
        print(f"\nSize: {size} points")
        print(f"  DFT:  {dft_time*1000:.2f} ms")
        print(f"  Sort: {sort_time*1000:.2f} ms")
        print(f"  IDFT: {idft_time*1000:.2f} ms")
        print(f"  Total: {(dft_time + sort_time + idft_time)*1000:.2f} ms")


def benchmark_image_preprocessing():
    """Benchmark image preprocessing operations."""
    print("\n=== Image Preprocessing Benchmark ===")
    
    processor = OpenCVImageProcessor()
    sizes = [(640, 480), (1280, 720), (1920, 1080), (2560, 1440)]
    
    for width, height in sizes:
        # Create synthetic image
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Benchmark preprocessing
        config = PreprocessConfig(
            target_size=(800, 600),
            maintain_aspect_ratio=True,
            normalize=True,
            denoise=False
        )
        
        start = time.perf_counter()
        processed = processor.preprocess(image, config)
        preprocess_time = time.perf_counter() - start
        
        print(f"\nResolution: {width}x{height}")
        print(f"  Preprocessing: {preprocess_time*1000:.2f} ms")


def benchmark_serialization():
    """Benchmark coefficient serialization/deserialization."""
    print("\n=== Serialization Benchmark ===")
    
    serializer = CoefficientSerializer()
    sizes = [50, 100, 500, 1000]
    
    for size in sizes:
        # Create synthetic coefficients with consistent amplitude/phase/complex_value
        coefficients = []
        for i in range(size):
            amplitude = float(np.random.random())
            phase = float(np.random.random() * 2 * np.pi - np.pi)
            # Reconstruct complex value from amplitude and phase
            complex_value = amplitude * complex(np.cos(phase), np.sin(phase))
            
            coefficients.append(
                FourierCoefficient(
                    frequency=i,
                    amplitude=amplitude,
                    phase=phase,
                    complex_value=complex_value
                )
            )
        
        metadata = {"dimensions": [1920, 1080]}
        
        # Benchmark serialization
        start = time.perf_counter()
        binary_data = serializer.serialize(coefficients, metadata)
        serialize_time = time.perf_counter() - start
        
        # Benchmark deserialization
        start = time.perf_counter()
        deserialized_coeffs, deserialized_meta = serializer.deserialize(binary_data)
        deserialize_time = time.perf_counter() - start
        
        print(f"\nCoefficients: {size}")
        print(f"  Serialize:   {serialize_time*1000:.2f} ms")
        print(f"  Deserialize: {deserialize_time*1000:.2f} ms")
        print(f"  Data size:   {len(binary_data)} bytes")
        print(f"  Bytes/coeff: {len(binary_data)/size:.1f}")


def benchmark_end_to_end():
    """Benchmark a complete encryption workflow."""
    print("\n=== End-to-End Workflow Benchmark ===")
    
    # Create synthetic data
    num_points = 1000
    points = np.random.random(num_points) + 1j * np.random.random(num_points)
    
    # Initialize components
    transformer = FourierTransformer()
    serializer = CoefficientSerializer()
    
    # Full workflow
    start_total = time.perf_counter()
    
    # 1. DFT
    start = time.perf_counter()
    coefficients = transformer.compute_dft(points)
    dft_time = time.perf_counter() - start
    
    # 2. Sort and truncate
    start = time.perf_counter()
    sorted_coeffs = transformer.sort_by_amplitude(coefficients)
    truncated = transformer.truncate_coefficients(sorted_coeffs, 500)
    sort_time = time.perf_counter() - start
    
    # 3. Serialize
    start = time.perf_counter()
    binary_data = serializer.serialize(truncated, {"dimensions": [1920, 1080]})
    serialize_time = time.perf_counter() - start
    
    # 4. Deserialize
    start = time.perf_counter()
    deserialized_coeffs, _ = serializer.deserialize(binary_data)
    deserialize_time = time.perf_counter() - start
    
    # 5. IDFT
    start = time.perf_counter()
    reconstructed = transformer.compute_idft(deserialized_coeffs)
    idft_time = time.perf_counter() - start
    
    total_time = time.perf_counter() - start_total
    
    print(f"\nWorkflow breakdown ({num_points} points → 500 coefficients):")
    print(f"  1. DFT:          {dft_time*1000:.2f} ms ({dft_time/total_time*100:.1f}%)")
    print(f"  2. Sort/Truncate: {sort_time*1000:.2f} ms ({sort_time/total_time*100:.1f}%)")
    print(f"  3. Serialize:    {serialize_time*1000:.2f} ms ({serialize_time/total_time*100:.1f}%)")
    print(f"  4. Deserialize:  {deserialize_time*1000:.2f} ms ({deserialize_time/total_time*100:.1f}%)")
    print(f"  5. IDFT:         {idft_time*1000:.2f} ms ({idft_time/total_time*100:.1f}%)")
    print(f"  Total:           {total_time*1000:.2f} ms")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("Performance Optimization Benchmarks")
    print("=" * 60)
    
    benchmark_dft_computation()
    benchmark_image_preprocessing()
    benchmark_serialization()
    benchmark_end_to_end()
    
    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
