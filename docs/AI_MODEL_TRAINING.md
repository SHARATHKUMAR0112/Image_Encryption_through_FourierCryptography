# AI Model Training Guide

## Overview

This guide provides comprehensive instructions for training custom AI models for the Fourier-Based Image Encryption System. The system supports three types of AI models:

1. **Edge Detection Models** - CNN or Vision Transformer models for improved edge detection
2. **Coefficient Optimizer Models** - Regression or RL models for optimal coefficient selection
3. **Anomaly Detection Models** - Classification models for tamper detection

## Prerequisites

### Software Requirements

- Python 3.9+
- PyTorch 2.0+ or TensorFlow 2.10+
- CUDA 11.8+ (for GPU training)
- NumPy, OpenCV, scikit-learn
- Weights & Biases (optional, for experiment tracking)

### Hardware Requirements

**Minimum:**
- 16GB RAM
- 4GB GPU VRAM
- 50GB storage

**Recommended:**
- 32GB+ RAM
- 8GB+ GPU VRAM (NVIDIA RTX 3070 or better)
- 200GB+ SSD storage

## 1. Edge Detection Model Training

### 1.1 Dataset Preparation

**Dataset Structure:**
```
edge_detection_dataset/
├── train/
│   ├── images/
│   │   ├── img_0001.png
│   │   ├── img_0002.png
│   │   └── ...
│   └── edges/
│       ├── img_0001.png  # Ground truth edge maps
│       ├── img_0002.png
│       └── ...
├── val/
│   ├── images/
│   └── edges/
└── test/
    ├── images/
    └── edges/
```

**Dataset Requirements:**
- Minimum 10,000 training images
- Diverse image types (portraits, objects, scenes)
- High-quality ground truth edge maps
- Image resolution: 256x256 to 1024x1024
- Formats: PNG, JPG

**Ground Truth Generation:**

Use multiple edge detectors and manual annotation:

```python
import cv2
import numpy as np

def generate_ground_truth(image_path, output_path):
    """Generate ground truth edge map using ensemble of detectors."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Canny edges
    canny = cv2.Canny(img, 50, 150)
    
    # Sobel edges
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = (sobel / sobel.max() * 255).astype(np.uint8)
    
    # Combine (weighted average)
    edges = cv2.addWeighted(canny, 0.7, sobel, 0.3, 0)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(output_path, edges)
```

### 1.2 Model Architecture

**Option 1: U-Net (Recommended for beginners)**

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    """U-Net architecture for edge detection."""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(b) + e4
        d3 = self.dec3(d4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1
        
        # Output
        out = self.sigmoid(self.out(d1))
        return out
```

**Option 2: Vision Transformer (Advanced)**

```python
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class ViTEdgeDetector(nn.Module):
    """Vision Transformer for edge detection."""
    
    def __init__(self, image_size=256, patch_size=16):
        super(ViTEdgeDetector, self).__init__()
        
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
        )
        
        self.vit = ViTModel(config)
        
        # Decoder head
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # ViT encoder
        outputs = self.vit(x)
        features = outputs.last_hidden_state
        
        # Reshape for decoder
        B, N, C = features.shape
        H = W = int((N - 1) ** 0.5)  # Exclude CLS token
        features = features[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Decode to edge map
        edges = self.decoder(features)
        return edges
```

### 1.3 Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

class EdgeDataset(Dataset):
    """Dataset for edge detection training."""
    
    def __init__(self, image_dir, edge_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.edge_dir = Path(edge_dir)
        self.image_files = sorted(list(self.image_dir.glob("*.png")))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        
        # Load edge map
        edge_path = self.edge_dir / img_path.name
        edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        edge = edge.astype(np.float32) / 255.0
        
        # Convert to tensors
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dim
        edge = torch.from_numpy(edge).unsqueeze(0)
        
        if self.transform:
            img = self.transform(img)
            edge = self.transform(edge)
        
        return img, edge

def train_edge_detector(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-4,
    device="cuda"
):
    """Train edge detection model."""
    
    model = model.to(device)
    criterion = nn.BCELoss()  # Binary cross-entropy for edge maps
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, edges in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            edges = edges.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, edges)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, edges in val_loader:
                images = images.to(device)
                edges = edges.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, edges)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_edge_detector.pth')
            print(f"Saved best model with val_loss = {val_loss:.4f}")

# Usage
if __name__ == "__main__":
    # Create datasets
    train_dataset = EdgeDataset("data/train/images", "data/train/edges")
    val_dataset = EdgeDataset("data/val/images", "data/val/edges")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Create model
    model = UNet(in_channels=1, out_channels=1)
    
    # Train
    train_edge_detector(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
```

### 1.4 Evaluation Metrics

```python
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_edge_detector(model, test_loader, device="cuda"):
    """Evaluate edge detection model."""
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, edges in test_loader:
            images = images.to(device)
            edges = edges.to(device)
            
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(edges.cpu().numpy())
    
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    # Calculate metrics
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return {"f1": f1, "precision": precision, "recall": recall}
```

### 1.5 Hyperparameter Tuning

**Key Hyperparameters:**
- Learning rate: 1e-4 to 1e-3
- Batch size: 8 to 32 (depends on GPU memory)
- Number of epochs: 50 to 200
- Optimizer: Adam or AdamW
- Loss function: BCE, Dice Loss, or combination

**Recommended Configuration:**
```python
config = {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 100,
    "optimizer": "Adam",
    "loss": "BCE",
    "scheduler": "ReduceLROnPlateau",
    "patience": 5,
}
```

### 1.6 Model Export

```python
def export_model(model, output_path="edge_detector_v1.pth"):
    """Export trained model for use with the system."""
    
    # Save model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__,
        'version': '1.0.0',
        'input_size': (1, 256, 256),
        'output_size': (1, 256, 256),
        'metrics': {
            'f1_score': 0.92,  # Replace with actual metrics
            'precision': 0.90,
            'recall': 0.94,
        },
        'training_date': '2026-02-15',
    }, output_path)
    
    print(f"Model exported to {output_path}")
```

## 2. Coefficient Optimizer Model Training

### 2.1 Dataset Preparation

**Dataset Structure:**
```
optimizer_dataset/
├── train/
│   ├── images/
│   └── metadata.json  # Contains optimal coefficient counts
├── val/
│   ├── images/
│   └── metadata.json
└── test/
    ├── images/
    └── metadata.json
```

**Metadata Format:**
```json
{
  "img_0001.png": {
    "optimal_coefficients": 150,
    "complexity": "medium",
    "rmse": 0.03,
    "image_features": {
      "edge_density": 0.25,
      "frequency_content": 0.65,
      "texture_complexity": 0.45
    }
  }
}
```

### 2.2 Feature Extraction

```python
import cv2
import numpy as np
from scipy import fftpack

def extract_image_features(image_path):
    """Extract features for coefficient optimization."""
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    features = {}
    
    # Edge density
    edges = cv2.Canny(img, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / edges.size
    
    # Frequency content (FFT)
    fft = fftpack.fft2(img)
    fft_shift = fftpack.fftshift(fft)
    magnitude = np.abs(fft_shift)
    features['frequency_content'] = np.mean(magnitude)
    
    # Texture complexity (Laplacian variance)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    features['texture_complexity'] = laplacian.var()
    
    # Contrast
    features['contrast'] = img.std()
    
    # Entropy
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    features['entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
    
    return features
```

### 2.3 Model Architecture

```python
import torch
import torch.nn as nn

class CoefficientOptimizer(nn.Module):
    """Regression model for optimal coefficient count prediction."""
    
    def __init__(self, input_dim=5, hidden_dims=[128, 64, 32]):
        super(CoefficientOptimizer, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Output: coefficient count
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### 2.4 Training Script

```python
def train_optimizer(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-3,
    device="cuda"
):
    """Train coefficient optimizer model."""
    
    model = model.to(device)
    criterion = nn.MSELoss()  # Mean squared error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
```

### 2.5 Evaluation Metrics

- **RMSE** (Root Mean Squared Error): < 10 coefficients
- **MAE** (Mean Absolute Error): < 5 coefficients
- **R² Score**: > 0.85

## 3. Anomaly Detection Model Training

### 3.1 Dataset Preparation

**Dataset Structure:**
```
anomaly_dataset/
├── train/
│   ├── normal/  # Valid coefficient sets
│   └── anomalous/  # Tampered coefficient sets
├── val/
│   ├── normal/
│   └── anomalous/
└── test/
    ├── normal/
    └── anomalous/
```

### 3.2 Anomaly Generation

```python
def generate_anomalies(coefficients):
    """Generate tampered coefficient sets for training."""
    
    anomalies = []
    
    # Type 1: Random amplitude modification
    tampered = coefficients.copy()
    idx = np.random.randint(0, len(tampered))
    tampered[idx]['amplitude'] *= np.random.uniform(0.5, 2.0)
    anomalies.append(('amplitude_tamper', tampered))
    
    # Type 2: Phase discontinuity
    tampered = coefficients.copy()
    idx = np.random.randint(0, len(tampered))
    tampered[idx]['phase'] += np.random.uniform(-np.pi, np.pi)
    anomalies.append(('phase_tamper', tampered))
    
    # Type 3: Frequency gap
    tampered = coefficients.copy()
    del tampered[len(tampered)//2]
    anomalies.append(('frequency_gap', tampered))
    
    # Type 4: Distribution violation
    tampered = coefficients.copy()
    for i in range(len(tampered)):
        tampered[i]['amplitude'] = np.random.uniform(0, 100)
    anomalies.append(('distribution_violation', tampered))
    
    return anomalies
```

### 3.3 Model Architecture

```python
class AnomalyDetector(nn.Module):
    """Binary classifier for anomaly detection."""
    
    def __init__(self, input_dim=1000, hidden_dims=[512, 256, 128]):
        super(AnomalyDetector, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Binary output
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

### 3.4 Evaluation Metrics

- **Accuracy**: > 95%
- **Precision**: > 95%
- **Recall**: > 95%
- **F1-Score**: > 0.95
- **False Positive Rate**: < 5%

## 4. Integration with System

### 4.1 Model Loading

```python
from fourier_encryption.ai.model_repository import ModelRepository

# Load trained model
repo = ModelRepository()
model = repo.load_model("edge_detector", version="1.0.0")
```

### 4.2 Plugin Integration

Create a plugin for your custom model:

```python
from fourier_encryption.plugins import AIModelPlugin, PluginMetadata

class CustomEdgeDetectorPlugin(AIModelPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom-edge-detector",
            version="1.0.0",
            author="Your Name",
            description="Custom trained edge detector",
            plugin_type="ai_model",
            dependencies={"torch": "2.0.0"}
        )
    
    def initialize(self, config):
        self.model = torch.load(config["model_path"])
        self.model.eval()
    
    def detect_edges(self, image):
        # Implement edge detection
        pass
```

## 5. Dataset Preprocessing Pipeline

### 5.1 Data Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation():
    """Get augmentation pipeline for training."""
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        
        # Color transformations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # Normalization
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    """Get augmentation pipeline for validation."""
    return A.Compose([
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
```

### 5.2 Data Quality Checks

```python
def validate_dataset(dataset_path):
    """Validate dataset quality before training."""
    
    issues = []
    
    # Check image-edge pair existence
    image_dir = Path(dataset_path) / "images"
    edge_dir = Path(dataset_path) / "edges"
    
    image_files = set(f.name for f in image_dir.glob("*.png"))
    edge_files = set(f.name for f in edge_dir.glob("*.png"))
    
    missing_edges = image_files - edge_files
    if missing_edges:
        issues.append(f"Missing edge maps for: {missing_edges}")
    
    # Check image dimensions
    for img_file in image_dir.glob("*.png"):
        img = cv2.imread(str(img_file))
        if img is None:
            issues.append(f"Corrupted image: {img_file.name}")
        elif img.shape[0] < 256 or img.shape[1] < 256:
            issues.append(f"Image too small: {img_file.name} ({img.shape})")
    
    # Check edge map quality
    for edge_file in edge_dir.glob("*.png"):
        edge = cv2.imread(str(edge_file), cv2.IMREAD_GRAYSCALE)
        if edge is None:
            issues.append(f"Corrupted edge map: {edge_file.name}")
        elif np.sum(edge > 0) / edge.size < 0.01:
            issues.append(f"Edge map too sparse: {edge_file.name}")
    
    if issues:
        print("Dataset validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("Dataset validation passed!")
    return True
```

### 5.3 Dataset Splitting

```python
from sklearn.model_selection import train_test_split
import shutil

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split dataset into train/val/test sets."""
    
    assert train_ratio + val_ratio + test_ratio == 1.0
    
    # Get all image files
    image_files = list(Path(source_dir).glob("images/*.png"))
    
    # Split
    train_files, temp_files = train_test_split(image_files, train_size=train_ratio, random_state=42)
    val_files, test_files = train_test_split(temp_files, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (Path(output_dir) / split / 'images').mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / split / 'edges').mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_file in files:
            # Copy image
            shutil.copy(img_file, Path(output_dir) / split_name / 'images' / img_file.name)
            
            # Copy corresponding edge map
            edge_file = Path(source_dir) / 'edges' / img_file.name
            shutil.copy(edge_file, Path(output_dir) / split_name / 'edges' / img_file.name)
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
```

## 6. Hyperparameter Tuning Guidelines

### 6.1 Grid Search

```python
from itertools import product

def grid_search_hyperparameters():
    """Perform grid search for optimal hyperparameters."""
    
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [1e-5, 1e-4, 1e-3],
        'batch_size': [8, 16, 32],
        'optimizer': ['Adam', 'AdamW'],
        'weight_decay': [0, 1e-5, 1e-4],
    }
    
    best_score = float('inf')
    best_params = None
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        
        print(f"\nTesting parameters: {params}")
        
        # Train model with these parameters
        model = UNet()
        score = train_and_evaluate(model, params)
        
        if score < best_score:
            best_score = score
            best_params = params
            print(f"New best score: {best_score:.4f}")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    return best_params
```

### 6.2 Bayesian Optimization

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

def bayesian_optimization():
    """Use Bayesian optimization for hyperparameter tuning."""
    
    # Define search space
    space = [
        Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform'),
        Integer(8, 32, name='batch_size'),
        Categorical(['Adam', 'AdamW'], name='optimizer'),
        Real(0, 1e-3, name='weight_decay', prior='log-uniform'),
    ]
    
    @use_named_args(space)
    def objective(**params):
        """Objective function to minimize."""
        model = UNet()
        val_loss = train_and_evaluate(model, params)
        return val_loss
    
    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=50,
        random_state=42,
        verbose=True
    )
    
    print(f"Best parameters: {result.x}")
    print(f"Best validation loss: {result.fun:.4f}")
    
    return dict(zip([s.name for s in space], result.x))
```

### 6.3 Learning Rate Finder

```python
import matplotlib.pyplot as plt

def find_learning_rate(model, train_loader, device='cuda', start_lr=1e-7, end_lr=10, num_iter=100):
    """Find optimal learning rate using LR range test."""
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=(end_lr/start_lr)**(1/num_iter))
    
    lrs = []
    losses = []
    
    model.train()
    for i, (images, edges) in enumerate(train_loader):
        if i >= num_iter:
            break
        
        images = images.to(device)
        edges = edges.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, edges)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # Update learning rate
        lr_scheduler.step()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    plt.savefig('lr_finder.png')
    plt.close()
    
    # Find optimal LR (steepest descent)
    gradients = np.gradient(losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]
    
    print(f"Suggested learning rate: {optimal_lr:.2e}")
    return optimal_lr
```

### 6.4 Recommended Hyperparameter Ranges

**Edge Detection Models:**
```python
edge_detection_config = {
    'learning_rate': {
        'min': 1e-5,
        'max': 1e-3,
        'recommended': 1e-4,
        'description': 'Lower for fine-tuning, higher for training from scratch'
    },
    'batch_size': {
        'min': 4,
        'max': 64,
        'recommended': 16,
        'description': 'Depends on GPU memory; larger is better if possible'
    },
    'num_epochs': {
        'min': 50,
        'max': 300,
        'recommended': 100,
        'description': 'Use early stopping to prevent overfitting'
    },
    'optimizer': {
        'options': ['Adam', 'AdamW', 'SGD'],
        'recommended': 'Adam',
        'description': 'Adam works well for most cases'
    },
    'weight_decay': {
        'min': 0,
        'max': 1e-3,
        'recommended': 1e-5,
        'description': 'L2 regularization strength'
    },
    'dropout': {
        'min': 0,
        'max': 0.5,
        'recommended': 0.2,
        'description': 'Dropout probability for regularization'
    },
}
```

**Coefficient Optimizer Models:**
```python
optimizer_config = {
    'learning_rate': {
        'min': 1e-4,
        'max': 1e-2,
        'recommended': 1e-3,
        'description': 'Higher LR works for regression tasks'
    },
    'batch_size': {
        'min': 32,
        'max': 256,
        'recommended': 64,
        'description': 'Larger batches for stable gradient estimates'
    },
    'hidden_dims': {
        'options': [[128, 64, 32], [256, 128, 64], [512, 256, 128, 64]],
        'recommended': [128, 64, 32],
        'description': 'Network architecture depth and width'
    },
}
```

**Anomaly Detection Models:**
```python
anomaly_config = {
    'learning_rate': {
        'min': 1e-5,
        'max': 1e-3,
        'recommended': 5e-4,
        'description': 'Moderate LR for classification'
    },
    'batch_size': {
        'min': 16,
        'max': 128,
        'recommended': 32,
        'description': 'Balance between speed and stability'
    },
    'class_weight': {
        'description': 'Weight for positive class (anomalous)',
        'recommended': 'auto',
        'note': 'Use class_weight="balanced" for imbalanced datasets'
    },
}
```

## 7. Model Export and Versioning

### 7.1 Export Format

```python
def export_model_for_production(
    model,
    output_path,
    model_name,
    version,
    metrics,
    training_config,
    input_shape,
    output_shape
):
    """Export model with complete metadata for production use."""
    
    export_data = {
        # Model weights
        'model_state_dict': model.state_dict(),
        
        # Model metadata
        'metadata': {
            'name': model_name,
            'version': version,
            'architecture': model.__class__.__name__,
            'framework': 'pytorch',
            'framework_version': torch.__version__,
            'training_date': datetime.now().isoformat(),
            'author': 'Your Name',
            'description': 'Model description',
        },
        
        # Model specifications
        'specifications': {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'input_dtype': 'float32',
            'output_dtype': 'float32',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        },
        
        # Performance metrics
        'metrics': metrics,
        
        # Training configuration
        'training_config': training_config,
        
        # Model requirements
        'requirements': {
            'torch': torch.__version__,
            'numpy': np.__version__,
            'opencv': cv2.__version__,
        },
    }
    
    # Save
    torch.save(export_data, output_path)
    
    # Also save ONNX format for interoperability
    onnx_path = output_path.replace('.pth', '.onnx')
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported to:")
    print(f"  PyTorch: {output_path}")
    print(f"  ONNX: {onnx_path}")
    
    return output_path
```

### 7.2 Model Versioning Strategy

```python
class ModelVersion:
    """Semantic versioning for AI models."""
    
    def __init__(self, major, minor, patch):
        self.major = major  # Breaking changes
        self.minor = minor  # New features, backward compatible
        self.patch = patch  # Bug fixes
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"
    
    @classmethod
    def from_string(cls, version_str):
        major, minor, patch = map(int, version_str.split('.'))
        return cls(major, minor, patch)
    
    def increment_major(self):
        """Increment for breaking changes (architecture change, input/output change)."""
        return ModelVersion(self.major + 1, 0, 0)
    
    def increment_minor(self):
        """Increment for new features (improved accuracy, new capabilities)."""
        return ModelVersion(self.major, self.minor + 1, 0)
    
    def increment_patch(self):
        """Increment for bug fixes (training bug fixes, minor improvements)."""
        return ModelVersion(self.major, self.minor, self.patch + 1)
```

### 7.3 Model Registry

```python
import json
from pathlib import Path

class ModelRegistry:
    """Registry for tracking trained models."""
    
    def __init__(self, registry_path='models/registry.json'):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
    
    def _load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name, version, model_path, metrics, notes=''):
        """Register a new model version."""
        
        if model_name not in self.registry:
            self.registry[model_name] = []
        
        entry = {
            'version': version,
            'path': str(model_path),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'notes': notes,
        }
        
        self.registry[model_name].append(entry)
        self._save_registry()
        
        print(f"Registered {model_name} v{version}")
    
    def get_latest_version(self, model_name):
        """Get the latest version of a model."""
        if model_name not in self.registry:
            return None
        
        versions = self.registry[model_name]
        return max(versions, key=lambda x: ModelVersion.from_string(x['version']))
    
    def list_models(self):
        """List all registered models."""
        for model_name, versions in self.registry.items():
            print(f"\n{model_name}:")
            for v in versions:
                print(f"  v{v['version']} - {v['timestamp']}")
                print(f"    Metrics: {v['metrics']}")
                print(f"    Path: {v['path']}")
```

## 8. Custom AI Model Integration

### 8.1 Plugin Template

Create a file `my_custom_model_plugin.py`:

```python
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np

from fourier_encryption.plugins.base_plugin import AIModelPlugin, PluginMetadata
from fourier_encryption.models.exceptions import AIModelError

class MyCustomEdgeDetectorPlugin(AIModelPlugin):
    """Custom edge detection model plugin."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-custom-edge-detector",
            version="1.0.0",
            author="Your Name",
            description="Custom trained edge detection model with improved accuracy",
            plugin_type="ai_model",
            model_type="edge_detector",
            dependencies={
                "torch": ">=2.0.0",
                "numpy": ">=1.20.0",
                "opencv-python": ">=4.5.0"
            }
        )
    
    def initialize(self, config: Dict[str, Any]):
        """Initialize the plugin with configuration."""
        try:
            model_path = Path(config.get("model_path", "models/my_edge_detector.pth"))
            
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model = self._build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            # Load configuration
            self.threshold = config.get("threshold", 0.5)
            self.input_size = config.get("input_size", (256, 256))
            
            print(f"Initialized {self.metadata.name} on {self.device}")
            
        except Exception as e:
            raise AIModelError(f"Failed to initialize plugin: {e}")
    
    def _build_model(self):
        """Build the model architecture."""
        # Import your custom model architecture
        from my_models import MyCustomUNet
        return MyCustomUNet(in_channels=1, out_channels=1)
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the input image.
        
        Args:
            image: Grayscale image as numpy array (H, W)
        
        Returns:
            Binary edge map as numpy array (H, W)
        """
        try:
            # Preprocess
            original_shape = image.shape
            image_resized = cv2.resize(image, self.input_size)
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(image_tensor)
            
            # Postprocess
            edge_map = output.squeeze().cpu().numpy()
            edge_map = (edge_map > self.threshold).astype(np.uint8) * 255
            edge_map = cv2.resize(edge_map, (original_shape[1], original_shape[0]))
            
            return edge_map
            
        except Exception as e:
            raise AIModelError(f"Edge detection failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Return performance metrics."""
        return {
            "f1_score": 0.94,
            "precision": 0.92,
            "recall": 0.96,
            "inference_time_ms": 45.0,
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 8.2 Plugin Registration

```python
from fourier_encryption.plugins import PluginRegistry

# Register your plugin
registry = PluginRegistry()
registry.register_plugin(MyCustomEdgeDetectorPlugin)

# Use your plugin
config = {
    "model_path": "models/my_edge_detector.pth",
    "threshold": 0.5,
    "input_size": (256, 256)
}

plugin = registry.get_plugin("my-custom-edge-detector")
plugin.initialize(config)

# Use in encryption workflow
from fourier_encryption.application.orchestrator import EncryptionOrchestrator

orchestrator = EncryptionOrchestrator(
    edge_detector=plugin,
    # ... other components
)
```

### 8.3 Plugin Configuration File

Create `config/plugins.yaml`:

```yaml
plugins:
  edge_detectors:
    - name: my-custom-edge-detector
      enabled: true
      config:
        model_path: models/my_edge_detector_v1.0.0.pth
        threshold: 0.5
        input_size: [256, 256]
        use_gpu: true
  
  coefficient_optimizers:
    - name: my-custom-optimizer
      enabled: true
      config:
        model_path: models/my_optimizer_v1.0.0.pth
        target_rmse: 0.03
  
  anomaly_detectors:
    - name: my-custom-anomaly-detector
      enabled: true
      config:
        model_path: models/my_anomaly_detector_v1.0.0.pth
        confidence_threshold: 0.95
```

## 9. Example Training Configurations

### 9.1 Quick Training (Development)

```yaml
# config/training_quick.yaml
training:
  name: edge_detector_quick
  mode: development
  
dataset:
  train_path: data/train_small
  val_path: data/val_small
  batch_size: 16
  num_workers: 2
  augmentation: basic

model:
  architecture: UNet
  in_channels: 1
  out_channels: 1
  base_filters: 32

optimizer:
  type: Adam
  learning_rate: 1e-4
  weight_decay: 1e-5

training_params:
  num_epochs: 20
  early_stopping_patience: 5
  save_frequency: 5

logging:
  use_wandb: false
  log_frequency: 10
```

### 9.2 Full Training (Production)

```yaml
# config/training_full.yaml
training:
  name: edge_detector_production
  mode: production
  
dataset:
  train_path: data/train_full
  val_path: data/val_full
  test_path: data/test_full
  batch_size: 32
  num_workers: 8
  augmentation: advanced
  cache_dataset: true

model:
  architecture: UNet
  in_channels: 1
  out_channels: 1
  base_filters: 64
  depth: 5
  dropout: 0.2

optimizer:
  type: AdamW
  learning_rate: 1e-4
  weight_decay: 1e-4
  betas: [0.9, 0.999]

scheduler:
  type: ReduceLROnPlateau
  mode: min
  factor: 0.5
  patience: 10
  min_lr: 1e-7

training_params:
  num_epochs: 200
  early_stopping_patience: 20
  save_frequency: 10
  gradient_clip: 1.0
  mixed_precision: true

logging:
  use_wandb: true
  wandb_project: fourier-encryption
  wandb_entity: your-team
  log_frequency: 50
  save_samples: true

evaluation:
  metrics: [f1_score, precision, recall, iou]
  save_predictions: true
  compare_with_baseline: true
```

### 9.3 Transfer Learning

```yaml
# config/training_transfer.yaml
training:
  name: edge_detector_transfer
  mode: transfer_learning
  
pretrained:
  model_path: models/pretrained_edge_detector.pth
  freeze_encoder: true
  freeze_epochs: 10

dataset:
  train_path: data/custom_domain/train
  val_path: data/custom_domain/val
  batch_size: 16
  num_workers: 4
  augmentation: domain_specific

model:
  architecture: UNet
  load_pretrained: true
  fine_tune_layers: [decoder, output]

optimizer:
  type: Adam
  learning_rate: 1e-5  # Lower LR for fine-tuning
  weight_decay: 1e-5

training_params:
  num_epochs: 50
  early_stopping_patience: 10
  warmup_epochs: 5
```

### 9.4 Hyperparameter Search

```yaml
# config/training_hpsearch.yaml
training:
  name: edge_detector_hpsearch
  mode: hyperparameter_search
  
search:
  method: bayesian  # grid, random, bayesian
  num_trials: 50
  metric: val_f1_score
  direction: maximize

search_space:
  learning_rate:
    type: loguniform
    low: 1e-5
    high: 1e-3
  
  batch_size:
    type: categorical
    choices: [8, 16, 32, 64]
  
  dropout:
    type: uniform
    low: 0.0
    high: 0.5
  
  weight_decay:
    type: loguniform
    low: 1e-6
    high: 1e-3

dataset:
  train_path: data/train
  val_path: data/val
  num_workers: 4

model:
  architecture: UNet
  base_filters: 64

training_params:
  num_epochs: 30
  early_stopping_patience: 5
```

## 10. Best Practices

### Training

1. **Use data augmentation** (rotation, flip, brightness, contrast)
2. **Monitor overfitting** (early stopping, regularization)
3. **Use learning rate scheduling** (ReduceLROnPlateau, CosineAnnealing)
4. **Save checkpoints regularly**
5. **Track experiments** (Weights & Biases, TensorBoard)
6. **Validate on diverse datasets** before deployment
7. **Use mixed precision training** for faster training on modern GPUs
8. **Implement gradient clipping** to prevent exploding gradients

### Evaluation

1. **Test on diverse datasets**
2. **Compare with baseline methods**
3. **Measure inference time**
4. **Test on different hardware** (CPU, GPU)
5. **Validate on real-world data**
6. **Calculate confidence intervals** for metrics
7. **Perform error analysis** on failure cases

### Deployment

1. **Optimize model** (quantization, pruning)
2. **Test integration** with system
3. **Document model** (architecture, metrics, limitations)
4. **Version control** models
5. **Monitor performance** in production
6. **Implement A/B testing** for model updates
7. **Set up model rollback** procedures

## 6. Troubleshooting

### Common Issues

**Issue**: Model not converging
- **Solution**: Reduce learning rate, increase batch size, check data quality
- **Debug steps**:
  1. Check if loss is decreasing at all
  2. Verify data loading (visualize batches)
  3. Try simpler model first
  4. Check for NaN/Inf in gradients
  5. Reduce learning rate by 10x

**Issue**: Overfitting
- **Solution**: Add dropout, use data augmentation, reduce model complexity
- **Debug steps**:
  1. Monitor train vs. validation loss gap
  2. Increase dropout rate (0.2 → 0.5)
  3. Add more data augmentation
  4. Reduce model capacity
  5. Increase weight decay

**Issue**: Slow training
- **Solution**: Use GPU, reduce batch size, optimize data loading
- **Debug steps**:
  1. Profile code to find bottlenecks
  2. Use `num_workers > 0` in DataLoader
  3. Enable pin_memory for GPU
  4. Use mixed precision training
  5. Reduce image resolution

**Issue**: Poor generalization
- **Solution**: Increase dataset size, use data augmentation, regularization
- **Debug steps**:
  1. Collect more diverse training data
  2. Implement cross-validation
  3. Test on completely different dataset
  4. Check for data leakage
  5. Analyze failure cases

**Issue**: Out of memory (OOM)
- **Solution**: Reduce batch size, use gradient accumulation, optimize model
- **Debug steps**:
  1. Reduce batch size by half
  2. Use gradient accumulation
  3. Enable gradient checkpointing
  4. Clear cache: `torch.cuda.empty_cache()`
  5. Use smaller model variant

**Issue**: Unstable training (loss spikes)
- **Solution**: Reduce learning rate, use gradient clipping, check data
- **Debug steps**:
  1. Implement gradient clipping (max_norm=1.0)
  2. Reduce learning rate
  3. Check for corrupted data samples
  4. Use learning rate warmup
  5. Switch to more stable optimizer (AdamW)

**Issue**: Model predictions are all the same
- **Solution**: Check class imbalance, adjust loss weights, verify data
- **Debug steps**:
  1. Check class distribution in dataset
  2. Use weighted loss function
  3. Verify data preprocessing
  4. Check model initialization
  5. Reduce learning rate

**Issue**: GPU not being utilized
- **Solution**: Move model and data to GPU, check CUDA installation
- **Debug steps**:
  1. Verify: `torch.cuda.is_available()`
  2. Check: `model.to('cuda')`
  3. Check: `data.to('cuda')`
  4. Update CUDA drivers
  5. Reinstall PyTorch with CUDA support

## 11. Complete Training Script Templates

### 11.1 Edge Detection Training Script

Create `scripts/train_edge_detector.py`:

```python
#!/usr/bin/env python3
"""
Complete training script for edge detection models.
Usage: python scripts/train_edge_detector.py --config config/training_full.yaml
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from datetime import datetime

from models.unet import UNet
from datasets.edge_dataset import EdgeDataset
from utils.metrics import calculate_metrics
from utils.visualization import save_prediction_samples


def parse_args():
    parser = argparse.ArgumentParser(description='Train edge detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config):
    """Create train and validation dataloaders."""
    
    train_dataset = EdgeDataset(
        image_dir=config['dataset']['train_path'] + '/images',
        edge_dir=config['dataset']['train_path'] + '/edges',
        augmentation='advanced' if config['dataset']['augmentation'] == 'advanced' else 'basic'
    )
    
    val_dataset = EdgeDataset(
        image_dir=config['dataset']['val_path'] + '/images',
        edge_dir=config['dataset']['val_path'] + '/edges',
        augmentation=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(config):
    """Create model based on config."""
    
    model_config = config['model']
    
    if model_config['architecture'] == 'UNet':
        model = UNet(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            base_filters=model_config.get('base_filters', 64),
            depth=model_config.get('depth', 4),
            dropout=model_config.get('dropout', 0.0)
        )
    else:
        raise ValueError(f"Unknown architecture: {model_config['architecture']}")
    
    return model


def create_optimizer(model, config):
    """Create optimizer based on config."""
    
    opt_config = config['optimizer']
    
    if opt_config['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_config['learning_rate'],
            weight_decay=opt_config.get('weight_decay', 0)
        )
    elif opt_config['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_config['learning_rate'],
            weight_decay=opt_config.get('weight_decay', 0),
            betas=opt_config.get('betas', (0.9, 0.999))
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['type']}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config."""
    
    if 'scheduler' not in config:
        return None
    
    sched_config = config['scheduler']
    
    if sched_config['type'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_config.get('mode', 'min'),
            factor=sched_config.get('factor', 0.5),
            patience=sched_config.get('patience', 10),
            min_lr=sched_config.get('min_lr', 1e-7)
        )
    elif sched_config['type'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training_params']['num_epochs']
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, edges in pbar:
        images = images.to(device)
        edges = edges.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, edges)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, edges)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, edges in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            edges = edges.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, edges)
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(edges.cpu())
    
    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = calculate_metrics(all_preds, all_targets)
    
    return total_loss / len(val_loader), metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if config.get('logging', {}).get('use_wandb', False):
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging'].get('wandb_entity'),
            name=config['training']['name'],
            config=config
        )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config['training_params'].get('mixed_precision', False) else None
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training_params']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training_params']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Metrics: {metrics}")
        
        # Log to wandb
        if config.get('logging', {}).get('use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **metrics
            })
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config['training_params']['save_frequency'] == 0:
            checkpoint_path = f"checkpoints/{config['training']['name']}_epoch_{epoch+1}.pth"
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = f"models/{config['training']['name']}_best.pth"
            Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics,
                'config': config,
            }, best_model_path)
            print(f"Saved best model: {best_model_path} (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training_params']['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    if config.get('logging', {}).get('use_wandb', False):
        wandb.finish()


if __name__ == '__main__':
    main()
```

### 11.2 Coefficient Optimizer Training Script

Create `scripts/train_coefficient_optimizer.py`:

```python
#!/usr/bin/env python3
"""
Training script for coefficient optimizer models.
Usage: python scripts/train_coefficient_optimizer.py --config config/optimizer_training.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

class CoefficientDataset(Dataset):
    """Dataset for coefficient optimization training."""
    
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
        self.features = self.data['features']
        self.targets = self.data['targets']
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor([self.targets[idx]])
        )


def train_coefficient_optimizer(config):
    """Train coefficient optimizer model."""
    
    # Load data
    train_dataset = CoefficientDataset(config['dataset']['train_path'])
    val_dataset = CoefficientDataset(config['dataset']['val_path'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    from models.coefficient_optimizer import CoefficientOptimizer
    model = CoefficientOptimizer(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0)
    )
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training_params']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        
        for features, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, 'models/coefficient_optimizer_best.pth')
    
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_coefficient_optimizer(config)
```

### 11.3 Anomaly Detector Training Script

Create `scripts/train_anomaly_detector.py`:

```python
#!/usr/bin/env python3
"""
Training script for anomaly detection models.
Usage: python scripts/train_anomaly_detector.py --config config/anomaly_training.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class AnomalyDataset(Dataset):
    """Dataset for anomaly detection training."""
    
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
        self.coefficients = self.data['coefficients']
        self.labels = self.data['labels']  # 0 = normal, 1 = anomalous
    
    def __len__(self):
        return len(self.coefficients)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.coefficients[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


def train_anomaly_detector(config):
    """Train anomaly detection model."""
    
    # Load data
    train_dataset = AnomalyDataset(config['dataset']['train_path'])
    val_dataset = AnomalyDataset(config['dataset']['val_path'])
    
    # Handle class imbalance
    pos_weight = torch.FloatTensor([config['training_params'].get('pos_weight', 1.0)])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    from models.anomaly_detector import AnomalyDetector
    model = AnomalyDetector(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    pos_weight = pos_weight.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer'].get('weight_decay', 0)
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training loop
    best_f1 = 0.0
    
    for epoch in range(config['training_params']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        
        for coeffs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            coeffs = coeffs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(coeffs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for coeffs, labels in val_loader:
                coeffs = coeffs.to(device)
                labels = labels.to(device)
                
                outputs = model(coeffs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                },
                'config': config,
            }, 'models/anomaly_detector_best.pth')
            print(f"  Saved best model (F1: {f1:.4f})")
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_anomaly_detector(config)
```

## 12. Model Evaluation and Testing

### 12.1 Comprehensive Evaluation Script

Create `scripts/evaluate_model.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive model evaluation script.
Usage: python scripts/evaluate_model.py --model models/edge_detector_best.pth --test-data data/test
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_edge_detector(model_path, test_data_path, output_dir='evaluation_results'):
    """Comprehensive evaluation of edge detection model."""
    
    # Load model
    checkpoint = torch.load(model_path)
    model = UNet()  # Replace with your model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load test data
    test_images = list(Path(test_data_path).glob('images/*.png'))
    
    all_preds = []
    all_targets = []
    inference_times = []
    
    # Evaluate
    with torch.no_grad():
        for img_path in test_images:
            # Load image and ground truth
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            edge_path = Path(test_data_path) / 'edges' / img_path.name
            edge_gt = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
            
            # Preprocess
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            
            # Inference
            import time
            start = time.time()
            output = model(img_tensor)
            inference_time = (time.time() - start) * 1000  # ms
            inference_times.append(inference_time)
            
            # Postprocess
            pred = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
            all_preds.append(pred.flatten())
            all_targets.append((edge_gt / 255).flatten())
    
    # Calculate metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    metrics = {
        'f1_score': f1_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds),
        'recall': recall_score(all_targets, all_preds),
        'avg_inference_time_ms': np.mean(inference_times),
        'std_inference_time_ms': np.std(inference_times),
    }
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='evaluation_results')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate_edge_detector(args.model, args.test_data, args.output_dir)
```

## 7. Resources

### Datasets

- **BSDS500**: Berkeley Segmentation Dataset (edge detection)
- **NYUDv2**: RGB-D dataset with edge annotations
- **Cityscapes**: Urban scene understanding dataset

### Papers

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Vision Transformer (ViT): An Image is Worth 16x16 Words
- HED: Holistically-Nested Edge Detection

### Tools

- PyTorch: https://pytorch.org/
- TensorFlow: https://www.tensorflow.org/
- Weights & Biases: https://wandb.ai/
- TensorBoard: https://www.tensorflow.org/tensorboard

## 8. Support

For questions or issues:
- Check the main documentation in `docs/`
- Review example plugins in `fourier_encryption/plugins/examples/`
- Open an issue on GitHub

