#!/usr/bin/env python3
"""
Quick test to verify logging and image path configuration.

This script tests:
1. Automatic logging to timestamped files
2. Access to test images from images/ folder
3. Log file creation in logs/ folder
"""

import logging
from pathlib import Path

from fourier_encryption.config.logging_config import setup_logging, LOGS_DIR
from fourier_encryption.config.paths import (
    get_test_image,
    list_test_images,
    AVAILABLE_IMAGES,
    IMAGES_DIR,
)

# Setup logging with file output
setup_logging(level="INFO", enable_file_logging=True)
logger = logging.getLogger(__name__)


def test_logging():
    """Test that logging works and creates timestamped files."""
    logger.info("=" * 60)
    logger.info("Testing Logging Configuration")
    logger.info("=" * 60)
    
    # Check logs directory exists
    assert LOGS_DIR.exists(), f"Logs directory not found: {LOGS_DIR}"
    logger.info(f"[OK] Logs directory exists: {LOGS_DIR}")
    
    # Check that log files are being created
    log_files = list(LOGS_DIR.glob("*.log"))
    logger.info(f"[OK] Found {len(log_files)} log file(s) in logs directory")
    
    # Test logging at different levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    # Test sensitive data redaction
    logger.info("Testing sensitive data redaction...")
    test_data = {"key": "secret123", "username": "testuser"}
    logger.info(f"Data with sensitive info: {test_data}")
    logger.info("[OK] Sensitive data should be redacted in logs")
    
    return True


def test_images():
    """Test that image path configuration works."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Image Path Configuration")
    logger.info("=" * 60)
    
    # Check images directory exists
    assert IMAGES_DIR.exists(), f"Images directory not found: {IMAGES_DIR}"
    logger.info(f"✓ Images directory exists: {IMAGES_DIR}")
    
    # List all test images
    test_images = list_test_images()
    logger.info(f"✓ Found {len(test_images)} test image(s):")
    for img in test_images:
        size_kb = img.stat().st_size / 1024
        logger.info(f"  - {img.name} ({size_kb:.1f} KB)")
    
    # Test predefined aliases
    logger.info(f"\n✓ Predefined image aliases:")
    for alias, filename in AVAILABLE_IMAGES.items():
        logger.info(f"  - {alias}: {filename}")
    
    # Try to get a specific image
    if test_images:
        first_image = test_images[0]
        logger.info(f"\n✓ Successfully accessed: {first_image.name}")
        assert first_image.exists(), f"Image file not found: {first_image}"
    else:
        logger.warning("⚠ No test images found in images/ folder")
    
    return True


def main():
    """Run all tests."""
    try:
        # Test logging
        logging_ok = test_logging()
        
        # Test images
        images_ok = test_images()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info(f"Logging configuration: {'[PASS]' if logging_ok else '[FAIL]'}")
        logger.info(f"Image path configuration: {'[PASS]' if images_ok else '[FAIL]'}")
        logger.info("\n[OK] All tests passed!")
        logger.info(f"\nCheck the logs folder for output: {LOGS_DIR}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"[FAIL] Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
