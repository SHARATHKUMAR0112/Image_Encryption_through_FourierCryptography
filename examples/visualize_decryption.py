"""
Visualize the decryption process with epicycle animation.

This script demonstrates how the visualization module integrates with
the encryption/decryption workflow. It shows how encrypted Fourier
coefficients can be animated during decryption.
"""

import math
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fourier_encryption.visualization.live_renderer import LiveRenderer, RenderObserver
from fourier_encryption.models.data_models import FourierCoefficient, Metrics
from fourier_encryption.core.epicycle_engine import EpicycleEngine


class DecryptionMetricsObserver(RenderObserver):
    """
    Observer that tracks decryption metrics.
    
    This demonstrates how you would integrate the visualization
    with the monitoring dashboard during actual decryption.
    """
    
    def __init__(self, total_coefficients: int):
        self.total_coefficients = total_coefficients
        self.frames_rendered = 0
        self.start_time = None
    
    def on_frame_rendered(self, state, frame_number):
        """Track decryption progress."""
        import time
        
        if self.start_time is None:
            self.start_time = time.time()
        
        self.frames_rendered += 1
        
        # Calculate metrics
        elapsed = time.time() - self.start_time
        fps = self.frames_rendered / elapsed if elapsed > 0 else 0
        progress = (frame_number / 300) * 100  # Assuming 300 total frames
        
        # Print metrics every 30 frames (once per second at 30 FPS)
        if frame_number % 30 == 0:
            print(f"Decryption Progress: {progress:.1f}% | "
                  f"FPS: {fps:.1f} | "
                  f"Elapsed: {elapsed:.1f}s")


def simulate_decrypted_coefficients():
    """
    Simulate coefficients that would come from decryption.
    
    In a real scenario, these would be:
    1. Loaded from an encrypted file
    2. Decrypted with the user's key
    3. Deserialized from binary format
    4. Passed to the visualization
    """
    print("Simulating decrypted Fourier coefficients...")
    print("(In real usage, these would come from encrypted payload)")
    print()
    
    # Simulate a complex shape with multiple harmonics
    coefficients = []
    
    # Add DC component (center offset)
    coefficients.append(
        FourierCoefficient(
            frequency=0,
            amplitude=120.0,
            phase=0.0,
            complex_value=120.0 + 0j
        )
    )
    
    # Add harmonics with decreasing amplitudes (typical of real images)
    for freq in range(1, 10):
        # Amplitude decreases with frequency (power law)
        amplitude = 80.0 / freq
        
        # Phase varies with frequency
        phase = (freq * math.pi / 3) % (2 * math.pi) - math.pi
        
        coefficients.append(
            FourierCoefficient(
                frequency=freq,
                amplitude=amplitude,
                phase=phase,
                complex_value=amplitude * complex(math.cos(phase), math.sin(phase))
            )
        )
    
    return coefficients


def main():
    """Demonstrate visualization during decryption."""
    print("=" * 70)
    print("Epicycle Visualization - Decryption Process")
    print("=" * 70)
    print()
    print("This demonstrates how the visualization module would be used")
    print("during the decryption process to show the image reconstruction.")
    print()
    
    # Step 1: Simulate getting coefficients from decryption
    coefficients = simulate_decrypted_coefficients()
    
    print(f"Decrypted {len(coefficients)} Fourier coefficients")
    print()
    
    # Display coefficient information
    print("Coefficient Summary:")
    print(f"  - Total coefficients: {len(coefficients)}")
    print(f"  - Frequency range: {coefficients[0].frequency} to {coefficients[-1].frequency}")
    print(f"  - Max amplitude: {max(c.amplitude for c in coefficients):.2f}")
    print(f"  - Min amplitude: {min(c.amplitude for c in coefficients):.2f}")
    print()
    
    # Step 2: Create epicycle engine
    print("Creating epicycle engine...")
    engine = EpicycleEngine(coefficients)
    print()
    
    # Step 3: Create renderer with observer
    print("Setting up visualization with metrics tracking...")
    renderer = LiveRenderer(backend="matplotlib")
    
    # Attach metrics observer
    observer = DecryptionMetricsObserver(total_coefficients=len(coefficients))
    renderer.attach_observer(observer)
    print()
    
    # Step 4: Configure animation
    print("Choose animation speed:")
    print("  1 = Slow (1x)")
    print("  2 = Normal (2x)")
    print("  3 = Fast (4x)")
    print("  4 = Very Fast (6x) - Recommended")
    print("  5 = Ultra Fast (10x)")
    print("  Or enter any custom speed (0.1 to 10.0)")
    print()
    
    speed_input = input("Enter speed [default: 6x]: ").strip()
    
    # Parse speed input
    speed_map = {"1": 1.0, "2": 2.0, "3": 4.0, "4": 6.0, "5": 10.0}
    
    if speed_input in speed_map:
        speed = speed_map[speed_input]
    elif speed_input == "":
        speed = 6.0  # Default to 6x
    else:
        try:
            speed = float(speed_input)
            if speed < 0.1 or speed > 10.0:
                print(f"Speed {speed} out of range [0.1, 10.0], using 6x")
                speed = 6.0
        except ValueError:
            print(f"Invalid input, using default 6x")
            speed = 6.0
    
    fps = 30
    duration = 10  # seconds
    num_frames = fps * duration
    
    print()
    print("Animation Configuration:")
    print(f"  - Speed: {speed}x")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {duration} seconds")
    print(f"  - Total frames: {num_frames}")
    print(f"  - Complete rotations: {speed}")
    print()
    
    print("Starting decryption visualization...")
    print("Watch the epicycles reconstruct the encrypted image!")
    print()
    print("Visual Guide:")
    print("  - Blue circles = Epicycles (rotating frequency components)")
    print("  - Red line = Reconstructed contour path")
    print("  - Frame info = Progress indicator (top-left)")
    print()
    print("Close the matplotlib window when done.")
    print("=" * 70)
    print()
    
    try:
        # Start the animation
        renderer.animate(engine, fps=fps, num_frames=num_frames, speed=1.0)
        
        print()
        print("=" * 70)
        print("Decryption visualization completed!")
        print(f"Total frames rendered: {observer.frames_rendered}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
