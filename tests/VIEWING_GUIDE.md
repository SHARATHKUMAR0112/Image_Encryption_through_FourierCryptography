# Guide: Viewing Decrypted Contour Files (.npy)

## Quick Summary

Your decrypted file `output/test_decrypt.npy` contains **50 contour points** in complex number format (x + iy).

**Visualization created**: `output/test_decrypt.png` ✓

## Method 1: Quick Visualization (Recommended)

Use the provided quick viewer script:

```bash
python view_contour_quick.py output/test_decrypt.npy
```

This creates a PNG image showing:
- Left plot: Connected contour line
- Right plot: Individual points with color gradient showing sequence

**Output**: `output/test_decrypt.png`

## Method 2: Interactive Visualization

Use the interactive viewer (opens matplotlib window):

```bash
python view_decrypted_contour.py output/test_decrypt.npy
```

This allows you to:
- Zoom and pan
- Save custom views
- Inspect individual points

## Method 3: Use Existing Demo Scripts

The project includes several demo scripts for visualization:

### A. Decrypt with Live Animation

```bash
python main.py decrypt -i output/test_encrypt.json -o output/animated.npy -k "StrongPassword123!" --visualize --speed 4.0
```

This shows the **epicycle animation** as it reconstructs the contour!

### B. Use Example Scripts

```bash
# Quick demo with visualization
python examples/quick_demo.py

# Fast demo (optimized)
python examples/fast_demo.py

# Interactive animation with speed control
python examples/interactive_animation.py

# Visualization demo
python examples/visualization_demo.py
```

## Method 4: Python Script

Create a custom viewer:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load the contour
contour = np.load('output/test_decrypt.npy')

# Extract x and y coordinates
x = contour.real
y = contour.imag

# Plot
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b-', linewidth=2)
plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Decrypted Contour')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.gca().invert_yaxis()
plt.savefig('my_contour.png', dpi=150)
plt.show()
```

## Method 5: Jupyter Notebook

```python
import numpy as np
import matplotlib.pyplot as plt

# Load and display
contour = np.load('output/test_decrypt.npy')
x, y = contour.real, contour.imag

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, y, 'b-', linewidth=2)
ax.scatter(x, y, c=range(len(x)), cmap='viridis', s=50)
ax.set_title(f'Decrypted Contour ({len(contour)} points)')
ax.axis('equal')
ax.invert_yaxis()
plt.colorbar(ax.collections[0], label='Point Index')
plt.show()
```

## Understanding the Data

### File Format
- **Format**: NumPy binary (.npy)
- **Data type**: complex128 (complex numbers)
- **Structure**: 1D array of complex numbers
- **Representation**: Each point is `x + iy` where:
  - `x` = real part (horizontal coordinate)
  - `y` = imaginary part (vertical coordinate)

### Your Decrypted File
```
File: output/test_decrypt.npy
Points: 50
X range: [82694.56, 82694.56]  # All points have same X (vertical line)
Y range: [192.32, 40572.57]
Format: Complex numbers
```

**Note**: Your contour appears to be a vertical line, which suggests it might be a simplified representation. For full image reconstruction, use more coefficients (e.g., 200-500).

## Viewing Options Summary

| Method | Speed | Interactive | Animation | Best For |
|--------|-------|-------------|-----------|----------|
| view_contour_quick.py | Fast | No | No | Quick preview |
| view_decrypted_contour.py | Medium | Yes | No | Detailed inspection |
| main.py decrypt --visualize | Slow | Yes | Yes | Full experience |
| examples/quick_demo.py | Fast | Yes | Yes | Demo/testing |
| Custom Python script | Fast | Optional | No | Custom analysis |

## Tips

1. **For better results**, use more coefficients:
   ```bash
   python main.py encrypt -i images/luffy2.png -o output/better.json -k "StrongPassword123!" -n 200
   ```

2. **See the animation**:
   ```bash
   python main.py decrypt -i output/better.json -o output/better.npy -k "StrongPassword123!" --visualize --speed 5.0
   ```

3. **Adjust animation speed**:
   - `--speed 1.0` = Normal speed
   - `--speed 5.0` = 5x faster (recommended)
   - `--speed 10.0` = Maximum speed

4. **View existing examples**:
   ```bash
   # Check the examples directory
   dir examples
   
   # Run any example
   python examples/quick_demo.py
   ```

## Troubleshooting

**Issue**: "File not found"
- Check the path: `dir output\test_decrypt.npy`
- Use absolute path if needed

**Issue**: "Module not found"
- Install dependencies: `pip install numpy matplotlib`

**Issue**: "Contour looks wrong"
- Try with more coefficients (200-500)
- Check if encryption used correct image

**Issue**: "Animation window freezes"
- Close the window to continue
- Use `view_contour_quick.py` for non-interactive viewing

## Next Steps

1. ✓ View the generated PNG: `output/test_decrypt.png`
2. Try with more coefficients for better reconstruction
3. Use `--visualize` flag to see the epicycle animation
4. Explore the example scripts in `examples/` directory
5. Check the documentation in `docs/` for more details

## Additional Resources

- **Visualization Guide**: `docs/VISUALIZATION_GUIDE.md`
- **Example Scripts**: `examples/README.md`
- **Main Documentation**: `README.md`
- **API Guide**: `docs/API_GUIDE.md`

---

**Quick Command Reference**:
```bash
# View contour (quick)
python view_contour_quick.py output/test_decrypt.npy

# Decrypt with animation
python main.py decrypt -i output/test_encrypt.json -o output/animated.npy -k "StrongPassword123!" --visualize --speed 5.0

# Run demo
python examples/quick_demo.py
```
