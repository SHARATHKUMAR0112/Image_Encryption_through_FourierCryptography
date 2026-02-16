"""
Interactive epicycle animation demonstration.

This script opens a matplotlib window showing the live epicycle animation.
You can watch the epicycles rotate and draw the trace path in real-time.
"""

import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fourier_encryption.visualization.live_renderer import LiveRenderer, RenderObserver
from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.core.epicycle_engine import EpicycleEngine


class AnimationObserver(RenderObserver):
    """Observer that prints animation progress."""
    
    def __init__(self):
        self.frame_count = 0
    
    def on_frame_rendered(self, state, frame_number):
        """Track frames."""
        self.frame_count += 1
        if frame_number % 30 == 0:  # Print every second at 30 FPS
            print(f"Rendered {frame_number} frames...")


def create_simple_circle():
    """Create coefficients for a simple circle."""
    return [
        FourierCoefficient(
            frequency=1,
            amplitude=100.0,
            phase=0.0,
            complex_value=100.0 + 0j
        )
    ]


def create_flower_pattern():
    """Create coefficients for a flower-like pattern."""
    return [
        FourierCoefficient(
            frequency=0,
            amplitude=100.0,
            phase=0.0,
            complex_value=100.0 + 0j
        ),
        FourierCoefficient(
            frequency=3,
            amplitude=50.0,
            phase=0.0,
            complex_value=50.0 + 0j
        ),
        FourierCoefficient(
            frequency=5,
            amplitude=25.0,
            phase=math.pi / 4,
            complex_value=25.0 * complex(math.cos(math.pi/4), math.sin(math.pi/4))
        )
    ]


def create_star_pattern():
    """Create coefficients for a star-like pattern."""
    return [
        FourierCoefficient(
            frequency=0,
            amplitude=80.0,
            phase=0.0,
            complex_value=80.0 + 0j
        ),
        FourierCoefficient(
            frequency=5,
            amplitude=60.0,
            phase=0.0,
            complex_value=60.0 + 0j
        ),
        FourierCoefficient(
            frequency=2,
            amplitude=30.0,
            phase=math.pi / 2,
            complex_value=30.0 * complex(math.cos(math.pi/2), math.sin(math.pi/2))
        )
    ]


def create_complex_shape():
    """Create coefficients for a complex shape."""
    coeffs = []
    
    # Add multiple harmonics with decreasing amplitudes
    base_amplitude = 120.0
    for freq in range(0, 8):
        amplitude = base_amplitude / (freq + 1)
        phase = (freq * math.pi / 4) % (2 * math.pi) - math.pi
        
        coeffs.append(
            FourierCoefficient(
                frequency=freq,
                amplitude=amplitude,
                phase=phase,
                complex_value=amplitude * complex(math.cos(phase), math.sin(phase))
            )
        )
    
    return coeffs


def main():
    """Run interactive animation."""
    print("=" * 70)
    print("Interactive Epicycle Animation")
    print("=" * 70)
    print("\nChoose a pattern to animate:")
    print("1. Simple Circle (1 epicycle)")
    print("2. Flower Pattern (3 epicycles)")
    print("3. Star Pattern (3 epicycles)")
    print("4. Complex Shape (8 epicycles)")
    print()
    
    choice = input("Enter your choice (1-4) [default: 2]: ").strip()
    
    if choice == "1":
        coeffs = create_simple_circle()
        title = "Simple Circle"
    elif choice == "3":
        coeffs = create_star_pattern()
        title = "Star Pattern"
    elif choice == "4":
        coeffs = create_complex_shape()
        title = "Complex Shape"
    else:
        coeffs = create_flower_pattern()
        title = "Flower Pattern"
    
    print(f"\nAnimating: {title}")
    print(f"Number of epicycles: {len(coeffs)}")
    print()
    
    # Ask user for speed preference
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
    
    # Create engine and renderer
    engine = EpicycleEngine(coeffs)
    renderer = LiveRenderer(backend="matplotlib")
    
    # Attach observer
    observer = AnimationObserver()
    renderer.attach_observer(observer)
    
    # Animation parameters
    fps = 30
    duration_seconds = 10
    num_frames = fps * duration_seconds
    
    print()
    print(f"Animation settings:")
    print(f"  - Speed: {speed}x")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {duration_seconds} seconds")
    print(f"  - Total frames: {num_frames}")
    print(f"  - Complete rotations: {speed}")
    print()
    print("Starting animation...")
    print("Close the matplotlib window to exit.")
    print()
    
    try:
        # This will open a matplotlib window with the live animation
        renderer.animate(engine, fps=fps, num_frames=num_frames, speed=speed)
        
        print(f"\nAnimation completed!")
        print(f"Total frames rendered: {observer.frame_count}")
        
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user.")
    except Exception as e:
        print(f"\nError during animation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
