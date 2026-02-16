# How to See the Live Epicycle Animation

## Quick Start (30 seconds)

Run this command to see the animation with speed control:

```bash
python examples/quick_demo.py
```

You'll be prompted to choose a speed (default is 6x - Very Fast).

**Or for instant ultra-fast animation:**

```bash
python examples/fast_demo.py
```

This runs at 10x speed automatically - the pattern completes in ~1 second!

## What You'll See

The animation window shows:

1. **Blue Circles** - These are the epicycles (rotating circles)
   - Each circle rotates at a different speed (frequency)
   - Larger circles = larger amplitude coefficients
   - Smaller circles = smaller amplitude coefficients
   - **Speed control**: Choose 1x to 10x to control rotation speed

2. **Blue Lines** - Connect the centers of the epicycles
   - Shows how epicycles are chained together
   - Each epicycle starts where the previous one ends

3. **Red Line** - The trace path being drawn
   - This is the actual shape being reconstructed
   - Accumulates as the animation progresses
   - Shows the final contour/sketch

4. **Red Dot** - Current drawing position
   - The tip of the last epicycle
   - Traces out the red path

5. **Info Box** (top-left corner)
   - Current frame number
   - Completion percentage
   - Updates in real-time

## Animation Speed Options

All demos now support configurable speed:

- **1x (Slow)**: 1 complete rotation in 10 seconds - See every detail
- **2x (Normal)**: 2 complete rotations - Comfortable viewing
- **4x (Fast)**: 4 complete rotations - Quick overview
- **6x (Very Fast)**: 6 complete rotations - **Recommended default**
- **10x (Ultra Fast)**: 10 complete rotations in 10 seconds (~1 second per cycle)
- **Custom**: Any value between 0.1 and 10.0

Higher speeds make the animation complete faster while still showing the full pattern formation.

## Available Demo Scripts

### 1. Quick Demo with Speed Control (Recommended)
```bash
python examples/quick_demo.py
```
- Interactive speed selection (1x to 10x)
- Default: 6x (Very Fast)
- Beautiful flower pattern
- 3 epicycles

### 2. Fast Demo (No Prompts)
```bash
python examples/fast_demo.py
```
- Runs at 10x speed automatically
- Pattern completes in ~1 second
- Perfect for quick testing
- No user input required

### 3. Interactive Demo
```bash
python examples/interactive_animation.py
```
- Choose from 4 different patterns
- Choose animation speed (1x to 10x)
- Options: circle, flower, star, or complex shape
- Full customization

### 4. Decryption Visualization
```bash
python examples/visualize_decryption.py
```
- Shows how visualization integrates with decryption
- Speed control included
- Simulates the decryption process
- Includes metrics tracking
- 10 epicycles, more complex pattern

### 5. Feature Demo (No Animation Window)
```bash
python examples/visualization_demo.py
```
- Tests all features programmatically
- Prints results to console
- No animation window opens
- Good for understanding the API

## Understanding the Animation

### How Epicycles Work

The animation demonstrates **Fourier Series** visually:

1. Each epicycle represents one term in the Fourier series
2. The epicycle rotates at a frequency (how fast it spins)
3. The epicycle has an amplitude (its radius)
4. The epicycle has a phase (where it starts)

When you combine all the rotating epicycles, the tip traces out the original shape!

### Mathematical Intuition

```
Final Point = Σ (amplitude_k × e^(i × frequency_k × t + phase_k))
```

Each epicycle adds a rotating vector, and the sum of all vectors traces the path.

## Customizing the Animation

### Change Animation Speed (Now Built-in!)

All interactive demos now prompt for speed. You can also modify scripts directly:

```python
renderer.animate(engine, fps=30, num_frames=300, speed=6.0)  # 6x faster (recommended)
renderer.animate(engine, fps=30, num_frames=300, speed=10.0) # 10x faster (ultra fast)
```

**Speed Guide:**
- `speed=1.0` - Slow, see every detail (1 rotation in 10 seconds)
- `speed=2.0` - Normal viewing speed
- `speed=4.0` - Fast overview
- `speed=6.0` - **Recommended** - Very fast, good balance
- `speed=10.0` - Ultra fast, completes in ~1 second per rotation

### Change Duration

Modify `num_frames`:

```python
renderer.animate(engine, fps=30, num_frames=600, speed=6.0)  # 20 seconds at 6x speed
renderer.animate(engine, fps=30, num_frames=150, speed=6.0)  # 5 seconds at 6x speed
```

### Change FPS (Smoothness)

Higher FPS = smoother animation (but more CPU intensive):

```python
renderer.animate(engine, fps=60, num_frames=600, speed=6.0)  # 60 FPS, very smooth
```

### Create Your Own Pattern

Add your own coefficients:

```python
import math
from fourier_encryption.models.data_models import FourierCoefficient

# Create a simple circle
my_coefficients = [
    FourierCoefficient(
        frequency=1,      # Rotates once per cycle
        amplitude=100.0,  # Radius of 100
        phase=0.0,        # Starts at angle 0
        complex_value=100.0 + 0j
    )
]

# Add more coefficients to create complex shapes!
```

## Integration with Encryption System

The visualization module is designed to integrate with the full encryption system:

### During Encryption
```python
# After computing Fourier coefficients from an image
coefficients = fourier_transformer.compute_dft(contour_points)

# Visualize before encryption (optional)
engine = EpicycleEngine(coefficients)
renderer = LiveRenderer()
renderer.animate(engine, fps=30, num_frames=300)
```

### During Decryption
```python
# After decrypting the payload
coefficients = deserialize(decrypted_data)

# Visualize the reconstruction
engine = EpicycleEngine(coefficients)
renderer = LiveRenderer()
renderer.animate(engine, fps=30, num_frames=300)
```

## Troubleshooting

### Window doesn't open
- Make sure matplotlib is installed: `pip install matplotlib`
- Check you're not in a headless environment
- Try: `python -i examples/quick_demo.py`

### Animation is choppy
- Reduce FPS: `fps=15`
- Reduce number of epicycles
- Close other applications

### "Module not found" error
- Make sure you're in the project root directory
- Install dependencies: `pip install -r requirements.txt`

## Next Steps

1. ✅ Run `python examples/quick_demo.py` to see the animation
2. Try different patterns with `python examples/interactive_animation.py`
3. Explore the code in `fourier_encryption/visualization/live_renderer.py`
4. Read the full documentation in `examples/README.md`
5. Integrate visualization with your encryption workflow

## Technical Details

- **Backend**: Matplotlib (PyQtGraph support planned)
- **Pattern**: Observer pattern for extensibility
- **FPS**: Configurable, default 30 FPS
- **Features**: Zoom, pan, frame info, real-time updates

## Questions?

Check out:
- `examples/README.md` - Detailed examples documentation
- `fourier_encryption/visualization/live_renderer.py` - Source code
- `.kiro/specs/fourier-image-encryption/design.md` - Design document

Enjoy watching the epicycles! 🎨✨
