"""
Demonstration of the LiveRenderer visualization module.

This script shows how to use the LiveRenderer to visualize epicycle
animations with the Observer pattern.
"""

import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fourier_encryption.visualization.live_renderer import LiveRenderer, RenderObserver
from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.core.epicycle_engine import EpicycleEngine


class ProgressObserver(RenderObserver):
    """Observer that prints progress updates."""
    
    def __init__(self, print_every: int = 10):
        self.print_every = print_every
    
    def on_frame_rendered(self, state, frame_number):
        """Print progress every N frames."""
        if frame_number % self.print_every == 0:
            print(f"Frame {frame_number}: time={state.time:.2f}, "
                  f"trace_point=({state.trace_point.real:.1f}, {state.trace_point.imag:.1f})")


def demo_simple_circle():
    """Demonstrate a simple circle animation with one epicycle."""
    print("\n=== Demo 1: Simple Circle ===")
    print("Creating a single epicycle that traces a circle...")
    
    # Create a single coefficient representing a circle
    coeffs = [
        FourierCoefficient(
            frequency=1,
            amplitude=100.0,
            phase=0.0,
            complex_value=100.0 + 0j
        )
    ]
    
    # Create engine and renderer
    engine = EpicycleEngine(coeffs)
    renderer = LiveRenderer(backend="matplotlib")
    
    # Attach progress observer
    observer = ProgressObserver(print_every=5)
    renderer.attach_observer(observer)
    
    # Render frames manually (without showing animation window)
    print("Rendering 30 frames...")
    for i, state in enumerate(engine.generate_animation_frames(30)):
        renderer.render_frame(state, i)
    
    print(f"Total trace points: {len(renderer._trace_points)}")


def demo_complex_shape():
    """Demonstrate a complex shape with multiple epicycles."""
    print("\n=== Demo 2: Complex Shape with Multiple Epicycles ===")
    print("Creating multiple epicycles that combine to form a complex shape...")
    
    # Create multiple coefficients with different frequencies and amplitudes
    coeffs = [
        FourierCoefficient(
            frequency=0,
            amplitude=150.0,
            phase=0.0,
            complex_value=150.0 + 0j
        ),
        FourierCoefficient(
            frequency=1,
            amplitude=80.0,
            phase=math.pi / 4,
            complex_value=80.0 * complex(math.cos(math.pi/4), math.sin(math.pi/4))
        ),
        FourierCoefficient(
            frequency=2,
            amplitude=40.0,
            phase=-math.pi / 3,
            complex_value=40.0 * complex(math.cos(-math.pi/3), math.sin(-math.pi/3))
        ),
        FourierCoefficient(
            frequency=3,
            amplitude=20.0,
            phase=math.pi / 2,
            complex_value=20.0 * complex(math.cos(math.pi/2), math.sin(math.pi/2))
        )
    ]
    
    # Create engine and renderer
    engine = EpicycleEngine(coeffs)
    renderer = LiveRenderer(backend="matplotlib")
    
    # Attach progress observer
    observer = ProgressObserver(print_every=10)
    renderer.attach_observer(observer)
    
    # Render frames
    print("Rendering 50 frames...")
    for i, state in enumerate(engine.generate_animation_frames(50)):
        renderer.render_frame(state, i)
        
        # Verify state has correct number of positions
        assert len(state.positions) == len(coeffs)
    
    print(f"Total trace points: {len(renderer._trace_points)}")
    print(f"Number of epicycles: {len(coeffs)}")


def demo_zoom_and_pan():
    """Demonstrate zoom and pan controls."""
    print("\n=== Demo 3: Zoom and Pan Controls ===")
    print("Demonstrating view transformation controls...")
    
    coeffs = [
        FourierCoefficient(
            frequency=1,
            amplitude=100.0,
            phase=0.0,
            complex_value=100.0 + 0j
        )
    ]
    
    engine = EpicycleEngine(coeffs)
    renderer = LiveRenderer(backend="matplotlib")
    
    print(f"Initial view scale: {renderer._view_scale}")
    print(f"Initial view center: {renderer._view_center}")
    
    # Zoom in
    renderer.zoom_in(2.0)
    print(f"After zoom in (2x): scale={renderer._view_scale}")
    
    # Pan
    renderer.pan(50, 30)
    print(f"After pan (50, 30): center={renderer._view_center}")
    
    # Zoom out
    renderer.zoom_out(1.5)
    print(f"After zoom out (1.5x): scale={renderer._view_scale}")
    
    # Render a frame with modified view
    state = engine.compute_state(0.0)
    renderer.render_frame(state, 0)
    
    print("View transformations applied successfully!")


def demo_observer_pattern():
    """Demonstrate the Observer pattern with multiple observers."""
    print("\n=== Demo 4: Observer Pattern ===")
    print("Demonstrating multiple observers receiving notifications...")
    
    class CountingObserver(RenderObserver):
        """Observer that counts frames."""
        def __init__(self, name):
            self.name = name
            self.count = 0
        
        def on_frame_rendered(self, state, frame_number):
            self.count += 1
    
    coeffs = [
        FourierCoefficient(
            frequency=1,
            amplitude=100.0,
            phase=0.0,
            complex_value=100.0 + 0j
        )
    ]
    
    engine = EpicycleEngine(coeffs)
    renderer = LiveRenderer(backend="matplotlib")
    
    # Attach multiple observers
    observer1 = CountingObserver("Observer 1")
    observer2 = CountingObserver("Observer 2")
    observer3 = CountingObserver("Observer 3")
    
    renderer.attach_observer(observer1)
    renderer.attach_observer(observer2)
    renderer.attach_observer(observer3)
    
    print(f"Attached {len(renderer.observers)} observers")
    
    # Render frames
    num_frames = 20
    for i, state in enumerate(engine.generate_animation_frames(num_frames)):
        renderer.render_frame(state, i)
    
    # Check observer counts
    print(f"{observer1.name} received {observer1.count} frames")
    print(f"{observer2.name} received {observer2.count} frames")
    print(f"{observer3.name} received {observer3.count} frames")
    
    # Detach one observer
    renderer.detach_observer(observer2)
    print(f"\nDetached {observer2.name}")
    
    # Render more frames
    for i, state in enumerate(engine.generate_animation_frames(10)):
        renderer.render_frame(state, i + num_frames)
    
    print(f"{observer1.name} received {observer1.count} frames (total)")
    print(f"{observer2.name} received {observer2.count} frames (stopped)")
    print(f"{observer3.name} received {observer3.count} frames (total)")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("LiveRenderer Visualization Demonstrations")
    print("=" * 60)
    
    try:
        demo_simple_circle()
        demo_complex_shape()
        demo_zoom_and_pan()
        demo_observer_pattern()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
