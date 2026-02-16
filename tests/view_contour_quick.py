#!/usr/bin/env python3
"""
Quick script to visualize decrypted contour without interactive display.

Usage:
    python view_contour_quick.py output/test_decrypt.npy
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
    
    # Check if contour is complex numbers or (x, y) pairs
    if np.iscomplexobj(contour):
        # Complex representation: x + iy
        x = contour.real
        y = contour.imag
        print("Format: Complex numbers (x + iy)")
    elif len(contour.shape) > 1 and contour.shape[1] == 2:
        # (x, y) pairs
        x = contour[:, 0]
        y = contour[:, 1]
        print("Format: (x, y) coordinate pairs")
    else:
        print(f"Unknown format: {contour.shape}")
        return
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  X mean: {x.mean():.2f}")
    print(f"  Y mean: {y.mean():.2f}")
    print(f"{'='*60}\n")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Contour as connected line
    ax1.plot(x, y, 'b-', linewidth=2, label='Contour', alpha=0.7)
    ax1.plot(x, y, 'b.', markersize=4, alpha=0.5)
    ax1.plot(x[0], y[0], 'go', markersize=12, label='Start', zorder=5)
    ax1.plot(x[-1], y[-1], 'ro', markersize=12, label='End', zorder=5)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Decrypted Contour (Connected)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.invert_yaxis()  # Invert Y axis to match image coordinates
    
    # Plot 2: Contour as scatter points with color gradient
    scatter = ax2.scatter(x, y, c=range(len(x)), cmap='viridis', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax2.plot(x[0], y[0], 'go', markersize=15, label='Start', zorder=5, markeredgecolor='black', markeredgewidth=2)
    ax2.plot(x[-1], y[-1], 'ro', markersize=15, label='End', zorder=5, markeredgecolor='black', markeredgewidth=2)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Decrypted Contour (Point Sequence)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.invert_yaxis()
    
    # Add colorbar for point sequence
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Point Index', rotation=270, labelpad=20, fontsize=11)
    
    # Add info text
    info_text = f"Points: {len(contour)}\nX: [{x.min():.1f}, {x.max():.1f}]\nY: [{y.min():.1f}, {y.max():.1f}]"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save the visualization
    output_path = Path(npy_path).with_suffix('.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path.absolute()}")
    
    plt.close()
    
    # Also print the first few points
    print("\nFirst 5 points:")
    for i in range(min(5, len(contour))):
        if np.iscomplexobj(contour):
            print(f"  Point {i}: ({contour[i].real:.2f}, {contour[i].imag:.2f})")
        else:
            print(f"  Point {i}: ({contour[i, 0]:.2f}, {contour[i, 1]:.2f})")
    
    print("\n✓ Visualization complete!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_contour_quick.py <path_to_npy_file>")
        print("\nExample:")
        print("  python view_contour_quick.py output/test_decrypt.npy")
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
