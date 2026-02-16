# Images and Logging Configuration

This document describes the automatic logging and test image configuration for the Fourier Image Encryption project.

## Summary of Changes

### 1. Automatic Timestamped Logging

All terminal output is now automatically logged to timestamped files in the `logs/` folder.

**Features:**
- Automatic log file creation with timestamps: `fourier_encryption_YYYYMMDD_HHMMSS.log`
- `latest.log` file always contains the most recent run
- Sensitive data (keys, passwords, tokens) automatically redacted
- Structured logging with timestamps, module names, and log levels

**Configuration:**
- Location: `fourier_encryption/config/logging_config.py`
- Logs directory: `logs/`
- Default log level: INFO
- Format: `YYYY-MM-DD HH:MM:SS - module.name - LEVEL - message`

**Usage:**
```python
from fourier_encryption.config.logging_config import setup_logging, get_logger

# Logging is automatically enabled on import
logger = get_logger(__name__)
logger.info("This will be logged to terminal and file")

# Customize logging
setup_logging(level="DEBUG", enable_file_logging=True)
```

### 2. Test Images Configuration

Test images are now centrally managed in the `images/` folder with easy access utilities.

**Available Images:**
- `luffy_1.png` - Anime character (25.2 KB)
- `luffy2.png` - Anime character (61.8 KB)
- `img1_prettylady.jpg` - Portrait (18.1 KB)
- `lady2.jpeg` - Portrait (171.0 KB)
- `lady3.jpg` - Portrait (37.5 KB)
- `img_2_malebpy.webp` - Portrait (42.0 KB)

**Configuration:**
- Location: `fourier_encryption/config/paths.py`
- Images directory: `images/`
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

**Usage:**
```python
from fourier_encryption.config.paths import get_test_image, list_test_images, AVAILABLE_IMAGES

# Get a specific image
image_path = get_test_image("luffy_1.png")

# Use predefined aliases
luffy_path = get_test_image(AVAILABLE_IMAGES["luffy1"])

# List all available images
all_images = list_test_images()
for img_path in all_images:
    print(f"Found: {img_path.name}")
```

### 3. Test Fixtures

New pytest fixtures for easy test image access:

```python
import pytest
from tests.fixtures import luffy_image, lady_image, all_test_images, temp_test_image

def test_with_luffy(luffy_image):
    """Test using the Luffy image fixture."""
    assert luffy_image.exists()
    # Use luffy_image path...

def test_with_all_images(all_test_images):
    """Test with all available images."""
    for img_path in all_test_images:
        # Process each image...
        pass

def test_with_synthetic_image(temp_test_image):
    """Test with a temporary synthetic image."""
    # temp_test_image is auto-created and cleaned up
    pass
```

### 4. Example Scripts

**`examples/use_test_images.py`** - Demonstrates:
- Accessing test images from the images folder
- Automatic logging to timestamped files
- Basic encryption/decryption workflow with real images

**`test_logging_and_images.py`** - Verification script that tests:
- Logging configuration and file creation
- Image path configuration and access
- Sensitive data redaction

## File Structure

```
project/
├── images/                              # Test images
│   ├── README.md                        # Image documentation
│   ├── luffy_1.png
│   ├── luffy2.png
│   ├── img1_prettylady.jpg
│   ├── lady2.jpeg
│   ├── lady3.jpg
│   └── img_2_malebpy.webp
│
├── logs/                                # Auto-generated log files
│   ├── README.md                        # Logging documentation
│   ├── fourier_encryption_YYYYMMDD_HHMMSS.log  # Timestamped logs
│   └── latest.log                       # Most recent run
│
├── fourier_encryption/
│   └── config/
│       ├── logging_config.py            # Logging configuration
│       └── paths.py                     # Path configuration
│
├── tests/
│   └── fixtures/
│       └── __init__.py                  # Test fixtures for images
│
├── examples/
│   └── use_test_images.py               # Demo script
│
└── test_logging_and_images.py           # Verification script
```

## Quick Start

### 1. Verify Setup

```bash
python test_logging_and_images.py
```

This will:
- Test logging configuration
- List all available test images
- Create timestamped log files in `logs/`

### 2. Run Example

```bash
python examples/use_test_images.py
```

This demonstrates:
- Using test images from the images folder
- Automatic logging to files
- Basic encryption/decryption workflow

### 3. Check Logs

```bash
# View latest log
cat logs/latest.log

# View specific timestamped log
cat logs/fourier_encryption_20260212_213505.log

# Follow logs in real-time
tail -f logs/latest.log
```

## Security Features

### Automatic Sensitive Data Redaction

The logging system automatically redacts sensitive information:

```python
logger.info(f"Encrypting with key: {key}")
# Logs: "Encrypting with key: ***REDACTED***"

data = {"key": "secret123", "password": "pass456", "username": "user"}
logger.info(f"Data: {data}")
# Logs: "Data: {'key': '***REDACTED***', 'password': '***REDACTED***', 'username': 'user'}"
```

**Redacted fields:**
- `key`, `password`, `secret`, `token`, `api_key`, `private_key`

## Adding New Images

To add new test images:

1. Copy image file to `images/` folder
2. Optionally add alias in `fourier_encryption/config/paths.py`:

```python
AVAILABLE_IMAGES = {
    "luffy1": "luffy_1.png",
    "luffy2": "luffy2.png",
    "my_new_image": "my_image.jpg",  # Add here
}
```

3. Image is automatically discovered by `list_test_images()`

## Maintenance

### Log Files

- Log files are never automatically deleted
- Consider periodically archiving or removing old logs
- Each log file is independent
- `latest.log` is overwritten on each run

### Test Images

- Keep images reasonably sized (< 5 MB recommended)
- Use standard formats: JPG, PNG, BMP, WEBP
- Images up to 4K resolution are supported

## Troubleshooting

### No Log Files Created

Check that logging is enabled:
```python
from fourier_encryption.config.logging_config import setup_logging
setup_logging(enable_file_logging=True)
```

### Image Not Found

List available images:
```python
from fourier_encryption.config.paths import list_test_images
print([img.name for img in list_test_images()])
```

### Unicode Errors in Logs (Windows)

The system uses ASCII-safe characters for compatibility. If you see Unicode errors, they're handled gracefully and don't affect functionality.

## Benefits

1. **Automatic Logging**: No manual log file management needed
2. **Timestamped Files**: Easy to track runs over time
3. **Security**: Sensitive data automatically redacted
4. **Centralized Images**: All test images in one location
5. **Easy Access**: Simple utilities for image and log access
6. **Test Integration**: Pytest fixtures for easy testing
7. **Documentation**: README files in both folders

## Next Steps

- Run `python test_logging_and_images.py` to verify setup
- Check `logs/latest.log` to see logging in action
- Try `python examples/use_test_images.py` for a demo
- Add your own test images to the `images/` folder
- Use the fixtures in your tests for easy image access
