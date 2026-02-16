#!/usr/bin/env python3
"""
Simple script to visualize decrypted contour from .npy file.

Usage:
    python view_decrypted_contour.py output/test_decrypt.npy
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def view_contour(npy_path: str):
    """Load and visualize the decrypted contour."""
    
    # Load the contour points
    contour = np.load(npy_path)
    
    print(f"\n{'='*60}")
    print(f"Decrypted Contour Visualization")
    print(f"{'='*60}")
    print(f"File: {npy_path}")
    print(f"Shape: {contour.shape}")
    print(f"Data type: {contour.dtype}")
    print(f"Number of points: {len(contour)}")
    print(f"{'='*60}\n")
    
    # Check if contour is complex numbers or (x, y) pairs
    if np.iscomplexobj(contour):
        # Complex representation: x + iy
        x = contour.real
        y = contour.imag
        print("Format: Complex numbers (x + iy)")
    elif contour.shape[1] == 2:
        # (x, y) pairs
        x = contour[:, 0]
        y = contour[:, 1]
        print("Format: (x, y) coordinate pairs")
    else:
        print(f"Unknown format: {contour.shape}")
        return
    
    # Print some statistics
    print(f"\nX range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Y range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Contour as connected line
    ax1.plot(x, y, 'b-', linewidth=2, label='Contour')
    ax1.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax1.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Decrypted Contour (Connected)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.invert_yaxis()  # Invert Y axis to match image coordinates
    
    # Plot 2: Contour as scatter points
    ax2.scatter(x, y, c=range(len(x)), cmap='viridis', s=50, alpha=0.6)
    ax2.plot(x[0], y[0], 'go', markersize=10, label='Start')
    ax2.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Decrypted Contour (Points)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.invert_yaxis()
    
    # Add colorbar for point sequence
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Point Index', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = Path(npy_path).with_suffix('.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    print("\n✓ Visualization complete!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_decrypted_contour.py <path_to_npy_file>")
        print("\nExample:")
        print("  python view_decrypted_contour.py output/test_decrypt.npy")
        sys.exit(1)
    
    npy_path = sys.argv[1]
    
    if not Path(npy_path).exists():
        print(f"Error: File not found: {npy_path}")
        sys.exit(1)
    
    try:
        view_contour(npy_path)
    except Exception as e:
        print(f"\nError viewing contour: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
