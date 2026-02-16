"""
Test fixtures and sample data.

Provides reusable test fixtures including test images from the images/ folder.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile

from fourier_encryption.config.paths import IMAGES_DIR, get_test_image, list_test_images


@pytest.fixture
def test_image_from_folder():
    """
    Fixture that provides a test image from the images/ folder.
    
    Returns the first available image in the images folder.
    """
    images = list_test_images()
    if not images:
        pytest.skip("No test images found in images/ folder")
    return images[0]


@pytest.fixture
def luffy_image():
    """Fixture that provides the Luffy test image."""
    try:
        return get_test_image("luffy_1.png")
    except FileNotFoundError:
        pytest.skip("luffy_1.png not found in images/ folder")


@pytest.fixture
def lady_image():
    """Fixture that provides a lady test image."""
    try:
        return get_test_image("img1_prettylady.jpg")
    except FileNotFoundError:
        pytest.skip("img1_prettylady.jpg not found in images/ folder")


@pytest.fixture
def all_test_images():
    """Fixture that provides all available test images."""
    return list_test_images()


@pytest.fixture
def temp_test_image():
    """
    Fixture that creates a temporary test image (white square on black background).
    
    Use this when you need a simple synthetic image for testing.
    """
    # Create a simple test image (white square on black background)
    image = np.zeros((200, 200), dtype=np.uint8)
    image[50:150, 50:150] = 255
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = Path(f.name)
        cv2.imwrite(str(temp_path), image)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()
