#!/usr/bin/env python3
"""
Plugin System Demonstration

This script demonstrates the plugin system for the Fourier-Based Image Encryption System.

Features demonstrated:
- Loading plugins from directories
- Registering custom encryption strategies
- Registering custom AI models
- Using plugins for encryption/decryption
- Using plugins for edge detection

Usage:
    python examples/plugin_system_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fourier_encryption.plugins import (
    PluginLoader,
    EncryptionPluginRegistry,
    AIModelPluginRegistry,
)
from fourier_encryption.plugins.examples.example_encryption_plugin import ExampleXOREncryptor
from fourier_encryption.plugins.examples.example_ai_plugin import (
    ExampleSobelEdgeDetector,
    ExampleSimpleOptimizer,
)


def demo_encryption_plugin():
    """Demonstrate encryption plugin usage."""
    print("\n" + "=" * 70)
    print("ENCRYPTION PLUGIN DEMONSTRATION")
    print("=" * 70)
    
    # Get encryption registry (singleton)
    registry = EncryptionPluginRegistry()
    
    # Register example plugin
    plugin = ExampleXOREncryptor()
    registry.register(plugin)
    
    print(f"\n✓ Registered plugin: {plugin.metadata.name} v{plugin.metadata.version}")
    print(f"  Description: {plugin.metadata.description}")
    print(f"  Author: {plugin.metadata.author}")
    
    # Initialize plugin
    config = {"kdf_iterations": 10000}
    registry.initialize_plugin(plugin.metadata.name, config)
    print(f"\n✓ Initialized plugin with config: {config}")
    
    # Use plugin for encryption
    print("\n--- Encryption Test ---")
    
    # Derive key from password
    password = "my_secure_password"
    salt = b"0" * 32  # In production, use os.urandom(32)
    key = plugin.derive_key(password, salt)
    print(f"✓ Derived key from password (length: {len(key)} bytes)")
    
    # Encrypt data
    plaintext = b"This is secret Fourier coefficient data!"
    print(f"\nPlaintext: {plaintext.decode()}")
    
    encrypted_payload = plugin.encrypt(plaintext, key)
    print(f"\n✓ Encrypted successfully")
    print(f"  Ciphertext length: {len(encrypted_payload.ciphertext)} bytes")
    print(f"  IV length: {len(encrypted_payload.iv)} bytes")
    print(f"  HMAC length: {len(encrypted_payload.hmac)} bytes")
    print(f"  Algorithm: {encrypted_payload.metadata['algorithm']}")
    
    # Decrypt data
    decrypted = plugin.decrypt(encrypted_payload, key)
    print(f"\n✓ Decrypted successfully")
    print(f"Decrypted: {decrypted.decode()}")
    
    # Verify round-trip
    assert decrypted == plaintext, "Decryption failed!"
    print("\n✓ Round-trip verification passed!")
    
    # Test wrong key detection
    print("\n--- Wrong Key Test ---")
    wrong_key = plugin.derive_key("wrong_password", salt)
    try:
        plugin.decrypt(encrypted_payload, wrong_key)
        print("✗ ERROR: Should have detected wrong key!")
    except Exception as e:
        print(f"✓ Correctly detected wrong key: {type(e).__name__}")
    
    # Cleanup
    registry.cleanup_all()
    print("\n✓ Plugin cleanup complete")


def demo_ai_model_plugin():
    """Demonstrate AI model plugin usage."""
    print("\n" + "=" * 70)
    print("AI MODEL PLUGIN DEMONSTRATION")
    print("=" * 70)
    
    # Get AI model registry (singleton)
    registry = AIModelPluginRegistry()
    
    # Register edge detector plugin
    edge_detector = ExampleSobelEdgeDetector()
    registry.register(edge_detector)
    
    print(f"\n✓ Registered plugin: {edge_detector.metadata.name} v{edge_detector.metadata.version}")
    print(f"  Description: {edge_detector.metadata.description}")
    print(f"  Model type: {edge_detector.model_type}")
    print(f"  GPU support: {edge_detector.supports_gpu}")
    
    # Initialize plugin
    config = {"threshold": 50}
    registry.initialize_plugin(edge_detector.metadata.name, config)
    print(f"\n✓ Initialized plugin with config: {config}")
    
    # Create test image (simple square)
    print("\n--- Edge Detection Test ---")
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[25:75, 25:75] = 255  # White square on black background
    print(f"✓ Created test image: {test_image.shape}")
    
    # Run edge detection
    edges = edge_detector.predict(test_image)
    print(f"✓ Edge detection complete")
    print(f"  Output shape: {edges.shape}")
    print(f"  Edge pixels: {np.sum(edges > 0)}")
    print(f"  Non-edge pixels: {np.sum(edges == 0)}")
    
    # Register optimizer plugin
    print("\n--- Coefficient Optimizer Test ---")
    optimizer = ExampleSimpleOptimizer()
    registry.register(optimizer)
    
    print(f"\n✓ Registered plugin: {optimizer.metadata.name} v{optimizer.metadata.version}")
    print(f"  Model type: {optimizer.model_type}")
    
    registry.initialize_plugin(optimizer.metadata.name, {})
    
    # Test optimization with different edge densities
    test_cases = [
        ("Low complexity", np.zeros((100, 100))),  # No edges
        ("Medium complexity", edges),  # Some edges
        ("High complexity", np.ones((100, 100)) * 255),  # All edges
    ]
    
    for name, test_data in test_cases:
        optimal_count = optimizer.predict(test_data)[0]
        print(f"  {name}: {int(optimal_count)} coefficients")
    
    # List all registered plugins
    print("\n--- Registered Plugins ---")
    all_plugins = registry.list_plugins()
    for plugin_meta in all_plugins:
        print(f"  • {plugin_meta.name} v{plugin_meta.version} ({plugin_meta.plugin_type})")
    
    # List by model type
    print("\n--- Edge Detectors ---")
    edge_detectors = registry.list_by_model_type("edge_detector")
    for detector in edge_detectors:
        print(f"  • {detector.metadata.name}")
    
    # Cleanup
    registry.cleanup_all()
    print("\n✓ Plugin cleanup complete")


def demo_plugin_loader():
    """Demonstrate automatic plugin loading."""
    print("\n" + "=" * 70)
    print("PLUGIN LOADER DEMONSTRATION")
    print("=" * 70)
    
    # Create plugin loader (fresh registries)
    loader = PluginLoader()
    
    # Load plugins from examples directory
    examples_dir = Path(__file__).parent.parent / "fourier_encryption" / "plugins" / "examples"
    
    print(f"\n✓ Discovering plugins in: {examples_dir}")
    
    # Discover plugins
    plugin_classes = loader.discover_plugins(examples_dir)
    print(f"✓ Found {len(plugin_classes)} plugin class(es)")
    
    for plugin_class in plugin_classes:
        print(f"  • {plugin_class.__name__}")
    
    # Load and register plugins (without auto-initialize to avoid conflicts)
    print("\n--- Loading and Registering Plugins ---")
    count = loader.load_and_register(examples_dir, auto_initialize=False)
    print(f"✓ Registered {count} plugin(s)")
    
    # List registered encryption plugins
    print("\n--- Encryption Plugins ---")
    enc_plugins = loader.encryption_registry.list_plugins()
    for plugin_meta in enc_plugins:
        print(f"  • {plugin_meta.name} v{plugin_meta.version}")
        print(f"    {plugin_meta.description}")
    
    # List registered AI model plugins
    print("\n--- AI Model Plugins ---")
    ai_plugins = loader.ai_model_registry.list_plugins()
    for plugin_meta in ai_plugins:
        print(f"  • {plugin_meta.name} v{plugin_meta.version}")
        print(f"    {plugin_meta.description}")
        print(f"    Type: {plugin_meta.plugin_type}")
    
    # Cleanup
    loader.cleanup()
    print("\n✓ Loader cleanup complete")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("FOURIER ENCRYPTION PLUGIN SYSTEM DEMO")
    print("=" * 70)
    print("\nThis demo showcases the extensible plugin architecture.")
    print("Plugins enable custom encryption algorithms and AI models.")
    
    try:
        # Run demonstrations
        demo_encryption_plugin()
        demo_ai_model_plugin()
        demo_plugin_loader()
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\n✓ All demonstrations completed successfully!")
        print("\nNext steps:")
        print("  1. Review plugin code in fourier_encryption/plugins/examples/")
        print("  2. Read PLUGIN_DEVELOPMENT.md for creating custom plugins")
        print("  3. Explore extension points for post-quantum crypto and custom AI models")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
