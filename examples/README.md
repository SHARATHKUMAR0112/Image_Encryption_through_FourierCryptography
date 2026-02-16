# Visualization Examples

This directory contains example scripts demonstrating the LiveRenderer visualization module.

## Quick Start - See the Animation Now!

### Option 1: Quick Demo with Speed Control (Recommended)
The fastest way to see the epicycle animation with customizable speed:

```bash
python examples/quick_demo.py
```

You'll be prompted to choose a speed:
- **1** = Slow (1x) - See every detail
- **2** = Normal (2x) - Comfortable viewing
- **3** = Fast (4x) - Quick overview
- **4** = Very Fast (6x) - **Recommended default**
- **5** = Ultra Fast (10x) - Complete in 1 second!
- Or enter any custom speed between 0.1 and 10.0

### Option 2: Super Fast Demo (No Prompts)
See the complete animation in ~1 second:

```bash
python examples/fast_demo.py
```

Runs at 10x speed automatically - perfect for quick testing!

### Option 3: Interactive Demo
Choose from different patterns AND speed:

```bash
python examples/interactive_animation.py
```

You'll be prompted to choose:
1. Pattern (circle, flower, star, or complex shape)
2. Animation speed (1x to 10x)

### Option 4: Decryption Visualization
See how visualization integrates with decryption:

```bash
python examples/visualize_decryption.py
```

Includes speed control and metrics tracking.

### Option 5: Non-Interactive Demo
See all features without opening windows:

```bash
python examples/visualization_demo.py
```

This runs through all the features and prints results to the console.

## What You'll See

When you run the interactive demos, a matplotlib window will open showing:

### Visual Elements
- **Epicycle Circles**: Blue circles of different sizes rotating at different frequencies
- **Connection Lines**: Blue lines connecting the centers of epicycles
- **Trace Path**: Red line showing the path being drawn by the tip of the last epicycle
- **Current Point**: Red dot at the current drawing position
- **Frame Info**: Box in top-left showing frame number and completion percentage

### Animation Speed

All interactive demos now support speed control:

- **1x (Slow)**: See every detail, good for understanding how epicycles work
- **2x (Normal)**: Comfortable viewing speed
- **4x (Fast)**: Quick overview of the pattern
- **6x (Very Fast)**: **Recommended default** - Good balance of speed and visibility
- **10x (Ultra Fast)**: Complete pattern in ~1 second
- **Custom**: Enter any value between 0.1 and 10.0

The speed parameter controls how many complete rotations happen during the animation. At 6x speed, the epicycles complete 6 full rotations in 10 seconds.

### How It Works
1. Each epicycle is a rotating circle in the complex plane
2. Each epicycle's center is at the tip of the previous epicycle
3. The final point traces out the shape as the epicycles rotate
4. Higher speeds make the epicycles rotate faster, completing more cycles
5. This is how Fourier series work - combining rotating circles to draw any shape!

## Controls

While the animation is running:
- **Close the window** to stop the animation
- The animation will loop automatically after completing

## Customization

You can modify the scripts to:
- Change the number of epicycles
- Adjust animation speed (0.1x to 10x)
- Change FPS (frames per second)
- Modify the coefficients to draw different shapes

### Example: Create Your Own Pattern

```python
from fourier_encryption.visualization.live_renderer import LiveRenderer
from fourier_encryption.models.data_models import FourierCoefficient
from fourier_encryption.core.epicycle_engine import EpicycleEngine
import math

# Define your coefficients
coefficients = [
    FourierCoefficient(
        frequency=1,
        amplitude=100.0,
        phase=0.0,
        complex_value=100.0 + 0j
    ),
    # Add more coefficients here...
]

# Create and run animation
engine = EpicycleEngine(coefficients)
renderer = LiveRenderer(backend="matplotlib")
renderer.animate(engine, fps=30, num_frames=300, speed=1.0)
```

## Technical Details

### Observer Pattern
The LiveRenderer uses the Observer pattern. You can attach observers to receive notifications when frames are rendered:

```python
from fourier_encryption.visualization.live_renderer import RenderObserver

class MyObserver(RenderObserver):
    def on_frame_rendered(self, state, frame_number):
        print(f"Frame {frame_number} rendered!")

observer = MyObserver()
renderer.attach_observer(observer)
```

### Zoom and Pan
You can control the view programmatically:

```python
renderer.zoom_in(2.0)      # Zoom in by 2x
renderer.zoom_out(1.5)     # Zoom out by 1.5x
renderer.pan(50, 30)       # Pan by (50, 30)
```

## Requirements

- Python 3.9+
- matplotlib
- numpy
- All dependencies from `requirements.txt`

## Troubleshooting

### "Backend is not available" error
Make sure matplotlib is installed:
```bash
pip install matplotlib
```

### Animation window doesn't open
- Check that you're not running in a headless environment
- Try running with `python -i examples/quick_demo.py` to keep the window open

### Animation is slow
- Reduce FPS: `renderer.animate(engine, fps=15, ...)`
- Reduce number of frames: `num_frames=150`
- Use fewer epicycles

## Next Steps

After seeing the visualization:
1. Try modifying the coefficients to create your own patterns
2. Integrate the visualization with the encryption workflow
3. Use the Observer pattern to track metrics during animation
4. Explore the full encryption pipeline in the main application

Enjoy watching the epicycles dance! 🎨


## Monitoring Dashboard Demo

### Option 6: Monitoring Dashboard
See real-time metrics tracking during encryption operations:

```bash
python examples/monitoring_demo.py
```

This demo shows:
- **Thread-safe metric updates** during simulated encryption
- **Real-time system metrics** (memory usage, processing time)
- **Terminal display mode** for console output
- **GUI mode (JSON)** for integration with web dashboards
- **Context manager usage** for automatic start/stop
- **Custom callback functions** for automatic metric updates

### What You'll See

The monitoring dashboard displays:

```
============================================================
  FOURIER ENCRYPTION MONITORING DASHBOARD
============================================================

Status:              ENCRYPTING
Progress:            50.0%

--- Processing Metrics ---
Coefficient Index:   50
Active Radius:       5.0000
Processing Time:     2.57s

--- Performance Metrics ---
FPS:                 30.0
Memory Usage:        102.7 MB

============================================================
```

### Using the Dashboard in Your Code

**Basic Usage:**
```python
from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard
from fourier_encryption.models.data_models import Metrics

# Create dashboard
dashboard = MonitoringDashboard(update_interval=0.1)

# Start background monitoring
dashboard.start_monitoring()

# Update metrics during processing
metrics = Metrics(
    current_coefficient_index=50,
    active_radius=7.5,
    progress_percentage=50.0,
    fps=30.0,
    processing_time=5.0,
    memory_usage_mb=150.0,
    encryption_status="encrypting"
)
dashboard.update_metrics(metrics)

# Display dashboard
print(dashboard.display(mode="terminal"))

# Stop monitoring
dashboard.stop_monitoring()
```

**Context Manager (Recommended):**
```python
with MonitoringDashboard(update_interval=0.1) as dashboard:
    # Process data...
    dashboard.update_metrics(metrics)
    print(dashboard.display(mode="terminal"))
# Automatically stops monitoring
```

**Custom Callback for Automatic Updates:**
```python
def get_current_metrics():
    # Fetch metrics from your processing pipeline
    return Metrics(...)

dashboard.set_update_callback(get_current_metrics)
dashboard.start_monitoring()
# Callback is invoked automatically every update_interval seconds
```

### Display Modes

**Terminal Mode** (default):
- Formatted text output for console display
- Shows all metrics in a readable format
- Perfect for CLI applications

**GUI Mode** (JSON):
- Structured JSON output
- Easy integration with web dashboards
- Machine-readable format

```python
# Terminal mode
print(dashboard.display(mode="terminal"))

# GUI mode (JSON)
json_data = dashboard.display(mode="gui")
```

### Thread Safety

The MonitoringDashboard is fully thread-safe:
- Multiple threads can update metrics concurrently
- Background monitoring thread doesn't block main thread
- Safe to use in multi-threaded encryption pipelines

### Integration with Encryption Pipeline

The dashboard integrates seamlessly with the encryption workflow:

```python
from fourier_encryption.application.orchestrator import EncryptionOrchestrator
from fourier_encryption.visualization.monitoring_dashboard import MonitoringDashboard

# Create orchestrator and dashboard
orchestrator = EncryptionOrchestrator(...)
dashboard = MonitoringDashboard()

# Set callback to fetch metrics from orchestrator
dashboard.set_update_callback(lambda: orchestrator.get_current_metrics())

# Start monitoring
dashboard.start_monitoring()

# Run encryption
encrypted = orchestrator.encrypt_image(image_path, key, config)

# Display final metrics
print(dashboard.display(mode="terminal"))

dashboard.stop_monitoring()
```
