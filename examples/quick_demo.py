"""
Quick demo - Watch epicycles draw a flower pattern!

Run this script to see the live animation in action.
A matplotlib window will open showing the epicycles rotating and drawing.
"""

import math
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fourier_encryption.visualization.live_renderer import LiveRenderer
from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.core.epicycle_engine import EpicycleEngine

# Create a beautiful flower pattern with 3 epicycles
coefficients = [
    # Center circle (DC component)
    FourierCoefficient(
        frequency=0,
        amplitude=100.0,
        phase=0.0,
        complex_value=100.0 + 0j
    ),
    # First harmonic (creates petals)
    FourierCoefficient(
        frequency=3,
        amplitude=50.0,
        phase=0.0,
        complex_value=50.0 + 0j
    ),
    # Second harmonic (adds detail)
    FourierCoefficient(
        frequency=5,
        amplitude=25.0,
        phase=math.pi / 4,
        complex_value=25.0 * complex(math.cos(math.pi/4), math.sin(math.pi/4))
    )
]

print("=" * 60)
print("Epicycle Animation - Flower Pattern")
print("=" * 60)
print(f"\nNumber of epicycles: {len(coefficients)}")
print("\nWatch the rotating circles (epicycles) combine to draw a flower!")
print("The blue circles rotate at different speeds.")
print("The red line shows the path being traced.")
print()

# Ask user for speed preference
print("Choose animation speed:")
print("  1 = Slow (1x)")
print("  2 = Normal (2x)")
print("  3 = Fast (4x) - Recommended")
print("  4 = Very Fast (6x)")
print("  5 = Ultra Fast (10x)")
print("  Or enter any number between 0.1 and 10.0")
print()

speed_choice = input("Enter speed [default: 4x]: ").strip()

# Parse speed choice
if speed_choice == "1":
    speed = 1.0
    speed_name = "Slow"
elif speed_choice == "2":
    speed = 2.0
    speed_name = "Normal"
elif speed_choice == "3":
    speed = 4.0
    speed_name = "Fast"
elif speed_choice == "5":
    speed = 10.0
    speed_name = "Ultra Fast"
elif speed_choice == "":
    speed = 6.0  # Default to 6x (Very Fast)
    speed_name = "Very Fast (default)"
else:
    try:
        speed = float(speed_choice)
        if speed < 0.1 or speed > 10.0:
            print(f"Speed {speed} out of range, using 6x")
            speed = 6.0
        speed_name = f"{speed}x"
    except ValueError:
        print(f"Invalid input, using default 6x")
        speed = 6.0
        speed_name = "Very Fast (default)"

# Calculate duration based on speed
fps = 30
base_frames = 300
num_frames = base_frames
duration = num_frames / fps

print()
print(f"Animation settings: {speed_name} ({speed}x speed)")
print(f"  - FPS: {fps}")
print(f"  - Duration: {duration:.1f} seconds")
print(f"  - Rotations: {speed} complete cycles")
print()
print("Close the window when done.")
print("=" * 60)

# Create the engine and renderer
engine = EpicycleEngine(coefficients)
renderer = LiveRenderer(backend="matplotlib")

# Start the animation (this opens a matplotlib window)
renderer.animate(
    engine,
    fps=fps,
    num_frames=num_frames,
    speed=speed
)

print("\nAnimation finished!")
