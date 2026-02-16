"""
Path configuration for the Fourier encryption system.

Defines standard paths for images, logs, and other resources.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Images directory - contains test images
IMAGES_DIR = PROJECT_ROOT / "images"

# Logs directory - contains timestamped log files
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
IMAGES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def get_test_image(filename: str) -> Path:
    """
    Get path to a test image in the images folder.
    
    Args:
        filename: Name of the image file
    
    Returns:
        Full path to the image file
    
    Raises:
        FileNotFoundError: If image doesn't exist
    """
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def list_test_images() -> list[Path]:
    """
    List all available test images in the images folder.
    
    Returns:
        List of paths to image files
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [
        img for img in IMAGES_DIR.iterdir()
        if img.is_file() and img.suffix.lower() in image_extensions
    ]


# Available test images
AVAILABLE_IMAGES = {
    "luffy1": "luffy_1.png",
    "luffy2": "luffy2.png",
    "lady1": "img1_prettylady.jpg",
    "lady2": "lady2.jpeg",
    "lady3": "lady3.jpg",
    "male": "img_2_malebpy.webp",
}
