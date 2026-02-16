# Test Images

This folder contains test images used for encryption/decryption demonstrations and testing.

## Available Images

1. **luffy_1.png** - Anime character image (Luffy)
2. **luffy2.png** - Another Luffy image
3. **img1_prettylady.jpg** - Portrait photo
4. **lady2.jpeg** - Portrait photo
5. **lady3.jpg** - Portrait photo
6. **img_2_malebpy.webp** - Male portrait photo

## Usage

### In Python Code

```python
from fourier_encryption.config.paths import get_test_image, list_test_images

# Get a specific image
image_path = get_test_image("luffy_1.png")

# List all available images
all_images = list_test_images()

# Use predefined aliases
from fourier_encryption.config.paths import AVAILABLE_IMAGES
luffy_path = get_test_image(AVAILABLE_IMAGES["luffy1"])
```

### In Tests

```python
import pytest
from tests.fixtures import luffy_image, lady_image, all_test_images

def test_with_luffy(luffy_image):
    # Use the luffy_image fixture
    assert luffy_image.exists()

def test_with_all_images(all_test_images):
    # Iterate through all test images
    for img_path in all_test_images:
        # Process each image
        pass
```

### In Examples

```bash
# Run the demo script that uses test images
python examples/use_test_images.py
```

## Adding New Images

To add new test images:

1. Copy your image file to this folder
2. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
3. Optionally add an alias in `fourier_encryption/config/paths.py`

## Notes

- All images in this folder are automatically discovered by `list_test_images()`
- Images are used for testing and demonstration purposes
- The encryption system supports images up to 4K resolution
