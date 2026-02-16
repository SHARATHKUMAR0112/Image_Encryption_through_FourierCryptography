"""
Example client for the Fourier Image Encryption API.

This script demonstrates how to use the API endpoints for encryption,
decryption, and visualization.
"""

import json
import sys
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth


# API Configuration
API_BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "secure_password_123"


def check_health():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ API server is healthy")
            return True
        else:
            print(f"✗ API server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server. Is it running?")
        print(f"  Start the server with: python run_api.py")
        return False
    except Exception as e:
        print(f"✗ Error checking health: {e}")
        return False


def encrypt_image(image_path: str, key: str, num_coefficients: int = None):
    """
    Encrypt an image using the API.
    
    Args:
        image_path: Path to image file
        key: Encryption password
        num_coefficients: Optional number of Fourier coefficients
        
    Returns:
        Encrypted data dictionary or None if failed
    """
    print(f"\nEncrypting image: {image_path}")
    print(f"Key: {key}")
    if num_coefficients:
        print(f"Coefficients: {num_coefficients}")
    
    try:
        # Prepare request
        auth = HTTPBasicAuth(USERNAME, PASSWORD)
        
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"key": key}
            if num_coefficients:
                data["num_coefficients"] = num_coefficients
            
            # Send request
            response = requests.post(
                f"{API_BASE_URL}/encrypt",
                auth=auth,
                files=files,
                data=data,
                timeout=30
            )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Encryption successful")
            print(f"  Size: {result['size_bytes']} bytes")
            print(f"  Dimensions: {result['encrypted_data']['metadata']['dimensions']}")
            return result["encrypted_data"]
        else:
            print(f"✗ Encryption failed: {response.status_code}")
            print(f"  Error: {response.json()}")
            return None
    
    except FileNotFoundError:
        print(f"✗ Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"✗ Error during encryption: {e}")
        return None


def decrypt_image(encrypted_data: dict, key: str):
    """
    Decrypt an encrypted payload using the API.
    
    Args:
        encrypted_data: Encrypted data dictionary from encrypt_image
        key: Decryption password
        
    Returns:
        Reconstructed points or None if failed
    """
    print(f"\nDecrypting image...")
    print(f"Key: {key}")
    
    try:
        # Prepare request
        auth = HTTPBasicAuth(USERNAME, PASSWORD)
        
        data = {
            "ciphertext": encrypted_data["ciphertext"],
            "iv": encrypted_data["iv"],
            "hmac": encrypted_data["hmac"],
            "metadata": json.dumps(encrypted_data["metadata"]),
            "key": key
        }
        
        # Send request
        response = requests.post(
            f"{API_BASE_URL}/decrypt",
            auth=auth,
            data=data,
            timeout=30
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Decryption successful")
            print(f"  Points: {result['point_count']}")
            print(f"  Dimensions: {result['dimensions']}")
            return result["reconstructed_points"]
        else:
            print(f"✗ Decryption failed: {response.status_code}")
            print(f"  Error: {response.json()}")
            return None
    
    except Exception as e:
        print(f"✗ Error during decryption: {e}")
        return None


def visualize_animation(encrypted_data: dict, key: str, num_frames: int = 100):
    """
    Generate animation data using the API.
    
    Args:
        encrypted_data: Encrypted data dictionary from encrypt_image
        key: Decryption password
        num_frames: Number of animation frames
        
    Returns:
        Animation data or None if failed
    """
    print(f"\nGenerating animation...")
    print(f"Frames: {num_frames}")
    
    try:
        # Prepare request
        auth = HTTPBasicAuth(USERNAME, PASSWORD)
        
        data = {
            "ciphertext": encrypted_data["ciphertext"],
            "iv": encrypted_data["iv"],
            "hmac": encrypted_data["hmac"],
            "metadata": json.dumps(encrypted_data["metadata"]),
            "key": key,
            "num_frames": num_frames
        }
        
        # Send request
        response = requests.post(
            f"{API_BASE_URL}/visualize",
            auth=auth,
            data=data,
            timeout=30
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Animation generated")
            print(f"  Frames: {result['animation']['frame_count']}")
            print(f"  Coefficients: {result['animation']['coefficient_count']}")
            return result["animation"]
        else:
            print(f"✗ Visualization failed: {response.status_code}")
            print(f"  Error: {response.json()}")
            return None
    
    except Exception as e:
        print(f"✗ Error during visualization: {e}")
        return None


def main():
    """Run example API client."""
    print("=" * 60)
    print("Fourier Image Encryption API - Example Client")
    print("=" * 60)
    
    # Check if server is running
    if not check_health():
        sys.exit(1)
    
    # Example usage
    print("\n" + "=" * 60)
    print("Example: Encrypt, Decrypt, and Visualize")
    print("=" * 60)
    
    # Check if test image exists
    test_image = "test_images/simple_shape.png"
    if not Path(test_image).exists():
        print(f"\n✗ Test image not found: {test_image}")
        print("  Please provide a test image or update the path")
        sys.exit(1)
    
    # Encryption
    encryption_key = "my_secret_key_123"
    encrypted_data = encrypt_image(test_image, encryption_key, num_coefficients=50)
    
    if not encrypted_data:
        print("\n✗ Encryption failed, cannot continue")
        sys.exit(1)
    
    # Decryption
    reconstructed_points = decrypt_image(encrypted_data, encryption_key)
    
    if not reconstructed_points:
        print("\n✗ Decryption failed")
        sys.exit(1)
    
    # Visualization
    animation_data = visualize_animation(encrypted_data, encryption_key, num_frames=50)
    
    if not animation_data:
        print("\n✗ Visualization failed")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("✓ All operations completed successfully!")
    print(f"  - Encrypted image with {len(reconstructed_points)} points")
    print(f"  - Generated {animation_data['frame_count']} animation frames")
    print(f"  - Used {animation_data['coefficient_count']} Fourier coefficients")
    
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("1. Try with different images")
    print("2. Experiment with different coefficient counts")
    print("3. Test with wrong decryption keys")
    print("4. Visualize the animation data in a frontend")


if __name__ == "__main__":
    main()
