"""
Fast Demo - See the animation complete in 1 second!

This runs at 10x speed so you can see the full pattern quickly.
Perfect for quick demonstrations or testing.
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

print("=" * 60)
print("FAST Epicycle Animation - 10x Speed!")
print("=" * 60)
print(f"\nNumber of epicycles: {len(coefficients)}")
print("Speed: 10x (Ultra Fast)")
print("Duration: ~1 second")
print()
print("Watch the epicycles complete 10 full rotations in 1 second!")
print()
print("Close the window when done.")
print("=" * 60)

# Create the engine and renderer
engine = EpicycleEngine(coefficients)
renderer = LiveRenderer(backend="matplotlib")

# Start the animation at 10x speed
renderer.animate(
    engine,
    fps=30,
    num_frames=300,
    speed=10.0  # 10x speed - completes in ~1 second!
)

print("\nAnimation finished!")
print("The pattern completed 10 full rotations!")
