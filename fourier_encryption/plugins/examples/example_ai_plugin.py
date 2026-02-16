"""
Example AI model plugin demonstrating the plugin interface.

This is a simple Sobel-based edge detector for demonstration purposes.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
from scipy import ndimage

from fourier_encryption.plugins.base_plugin import AIModelPlugin, PluginMetadata
from fourier_encryption.models.exceptions import AIModelError


class ExampleSobelEdgeDetector(AIModelPlugin):
    """
    Example edge detection plugin using Sobel operator.
    
    This plugin demonstrates:
    - AI model plugin interface
    - Model initialization and cleanup
    - Inference implementation
    - Input/output shape specification
    
    Uses classical Sobel operator instead of deep learning for simplicity.
    """
    
    def __init__(self):
        """Initialize the plugin."""
        self._initialized = False
        self.threshold = 50
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="example-sobel-detector",
            version="1.0.0",
            author="Fourier Encryption Team",
            description="Example Sobel edge detection plugin",
            plugin_type="ai_model",
            dependencies={"scipy": ">=1.9.0", "numpy": ">=1.24.0"}
        )
    
    @property
    def model_type(self) -> str:
        """Return model type."""
        return "edge_detector"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Configuration dictionary
                - threshold: Edge detection threshold (default: 50)
        """
        self.threshold = config.get("threshold", 50)
        self._initialized = True
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self._initialized = False
    
    def load_model(self, model_path: Path) -> None:
        """
        Load model from disk.
        
        For this example, no model file is needed (Sobel is algorithmic).
        
        Args:
            model_path: Path to model file (unused for Sobel)
        """
        # Sobel operator doesn't require loading a model file
        pass
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run edge detection using Sobel operator.
        
        Args:
            input_data: Input image (grayscale, shape: (H, W))
            
        Returns:
            Binary edge map (shape: (H, W), values: 0 or 255)
            
        Raises:
            AIModelError: If inference fails
        """
        if not self._initialized:
            raise AIModelError("Plugin not initialized")
        
        try:
            # Ensure input is 2D grayscale
            if input_data.ndim != 2:
                if input_data.ndim == 3:
                    # Convert RGB to grayscale
                    input_data = np.mean(input_data, axis=2).astype(np.uint8)
                else:
                    raise AIModelError(f"Invalid input shape: {input_data.shape}")
            
            # Apply Sobel operator in X and Y directions
            sobel_x = ndimage.sobel(input_data, axis=0)
            sobel_y = ndimage.sobel(input_data, axis=1)
            
            # Compute gradient magnitude
            magnitude = np.hypot(sobel_x, sobel_y)
            
            # Normalize to 0-255 range
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
            
            # Apply threshold to get binary edge map
            edges = (magnitude > self.threshold).astype(np.uint8) * 255
            
            return edges
            
        except Exception as e:
            raise AIModelError(f"Edge detection failed: {e}")
    
    def get_input_shape(self) -> tuple:
        """
        Return expected input shape.
        
        Returns:
            Tuple (H, W) - accepts any size
        """
        return (-1, -1)  # Variable size
    
    def get_output_shape(self) -> tuple:
        """
        Return expected output shape.
        
        Returns:
            Tuple (H, W) - same as input
        """
        return (-1, -1)  # Same as input
    
    @property
    def supports_gpu(self) -> bool:
        """
        Check if model supports GPU acceleration.
        
        Returns:
            False - Sobel operator runs on CPU
        """
        return False


class ExampleSimpleOptimizer(AIModelPlugin):
    """
    Example coefficient optimizer plugin.
    
    Uses a simple heuristic based on image complexity:
    - Low complexity: 50 coefficients
    - Medium complexity: 200 coefficients
    - High complexity: 500 coefficients
    """
    
    def __init__(self):
        """Initialize the plugin."""
        self._initialized = False
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="example-simple-optimizer",
            version="1.0.0",
            author="Fourier Encryption Team",
            description="Example coefficient optimizer plugin",
            plugin_type="ai_model",
            dependencies={"numpy": ">=1.24.0"}
        )
    
    @property
    def model_type(self) -> str:
        """Return model type."""
        return "optimizer"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        self._initialized = True
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self._initialized = False
    
    def load_model(self, model_path: Path) -> None:
        """Load model (not needed for this simple optimizer)."""
        pass
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Predict optimal coefficient count based on image complexity.
        
        Args:
            input_data: Image features (edge density, variance, etc.)
            
        Returns:
            Array with single value: optimal coefficient count
        """
        if not self._initialized:
            raise AIModelError("Plugin not initialized")
        
        try:
            # Simple heuristic: use edge density as complexity measure
            edge_density = np.mean(input_data > 0)
            
            if edge_density < 0.1:
                # Low complexity
                optimal_count = 50
            elif edge_density < 0.3:
                # Medium complexity
                optimal_count = 200
            else:
                # High complexity
                optimal_count = 500
            
            return np.array([optimal_count])
            
        except Exception as e:
            raise AIModelError(f"Optimization failed: {e}")
    
    def get_input_shape(self) -> tuple:
        """Return expected input shape."""
        return (-1, -1)  # Variable size edge map
    
    def get_output_shape(self) -> tuple:
        """Return expected output shape."""
        return (1,)  # Single value: coefficient count
    
    @property
    def supports_gpu(self) -> bool:
        """Check GPU support."""
        return False
