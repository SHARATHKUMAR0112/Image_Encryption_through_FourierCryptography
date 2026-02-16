"""
Example usage of CoefficientOptimizer for intelligent coefficient selection.

This example demonstrates how to use the AI-based coefficient optimizer to
automatically determine the optimal number of Fourier coefficients needed
for image reconstruction while minimizing payload size.
"""

import numpy as np

from fourier_encryption.ai.coefficient_optimizer import CoefficientOptimizer
from fourier_encryption.core.contour_extractor import ContourExtractor
from fourier_encryption.core.fourier_transformer import FourierTransformer
from fourier_encryption.models.data_models import Contour


def main():
    """Demonstrate coefficient optimization."""
    
    print("=" * 70)
    print("Coefficient Optimizer Example")
    print("=" * 70)
    
    # Create a sample contour (circular shape)
    print("\n1. Creating sample contour (circle)...")
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    x = 100 + 50 * np.cos(t)
    y = 100 + 50 * np.sin(t)
    points = np.column_stack([x, y])
    
    contour = Contour(points=points, is_closed=True, length=len(points))
    print(f"   Created contour with {len(points)} points")
    
    # Convert to complex plane
    print("\n2. Converting to complex plane...")
    extractor = ContourExtractor()
    complex_points = extractor.to_complex_plane(contour)
    print(f"   Converted to {len(complex_points)} complex points")
    
    # Compute Fourier coefficients
    print("\n3. Computing Fourier coefficients...")
    transformer = FourierTransformer()
    coefficients = transformer.compute_dft(complex_points)
    print(f"   Computed {len(coefficients)} Fourier coefficients")
    
    # Create optimizer
    print("\n4. Creating coefficient optimizer...")
    optimizer = CoefficientOptimizer(
        target_error=0.05,  # 5% maximum reconstruction error
        min_coefficients=10,
        max_coefficients=1000
    )
    print("   Optimizer configured:")
    print(f"   - Target error: {optimizer.target_error * 100}%")
    print(f"   - Coefficient range: [{optimizer.min_coefficients}, {optimizer.max_coefficients}]")
    
    # Optimize coefficient count
    print("\n5. Optimizing coefficient count...")
    result = optimizer.optimize_count(coefficients, complex_points)
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Original coefficient count: {len(coefficients)}")
    print(f"Optimal coefficient count:  {result.optimal_count}")
    print(f"Size reduction:             {(1 - result.optimal_count / len(coefficients)) * 100:.1f}%")
    print(f"Complexity class:           {result.complexity_class}")
    print(f"Reconstruction error:       {result.reconstruction_error:.6f} ({result.reconstruction_error * 100:.4f}%)")
    print(f"\nExplanation: {result.explanation}")
    
    # Verify reconstruction quality
    print("\n6. Verifying reconstruction quality...")
    sorted_coeffs = transformer.sort_by_amplitude(coefficients)
    optimized_coeffs = sorted_coeffs[:result.optimal_count]
    
    # Pad coefficients for reconstruction
    padded_coeffs = optimizer._pad_coefficients(optimized_coeffs, len(coefficients))
    
    # Reconstruct signal
    reconstructed = transformer.compute_idft(padded_coeffs)
    
    # Compute actual error
    actual_error = optimizer.compute_reconstruction_error(complex_points, reconstructed)
    print(f"   Actual reconstruction error: {actual_error:.6f}")
    print(f"   Target error threshold:      {optimizer.target_error:.6f}")
    print(f"   ✓ Error within threshold: {actual_error <= optimizer.target_error}")
    
    # Demonstrate with different complexity levels
    print("\n" + "=" * 70)
    print("TESTING DIFFERENT SIGNAL COMPLEXITIES")
    print("=" * 70)
    
    test_signals = [
        ("Simple circle", np.exp(1j * np.linspace(0, 2 * np.pi, 100, endpoint=False))),
        ("Circle + 1 harmonic", np.exp(1j * np.linspace(0, 2 * np.pi, 100, endpoint=False)) + 
                                0.3 * np.exp(1j * 3 * np.linspace(0, 2 * np.pi, 100, endpoint=False))),
        ("Multiple harmonics", sum((1.0 / (i + 1)) * np.exp(1j * i * np.linspace(0, 2 * np.pi, 150, endpoint=False))
                                   for i in range(1, 10))),
    ]
    
    for name, signal in test_signals:
        coeffs = transformer.compute_dft(signal)
        result = optimizer.optimize_count(coeffs, signal)
        
        print(f"\n{name}:")
        print(f"  Original: {len(coeffs)} coefficients")
        print(f"  Optimized: {result.optimal_count} coefficients ({(1 - result.optimal_count / len(coeffs)) * 100:.1f}% reduction)")
        print(f"  Complexity: {result.complexity_class}")
        print(f"  Error: {result.reconstruction_error:.6f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
