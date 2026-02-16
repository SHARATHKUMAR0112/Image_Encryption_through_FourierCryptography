# Training Configuration Files

This directory contains example training configurations for all AI models in the Fourier-Based Image Encryption System.

## Available Configurations

### Edge Detection Models

1. **edge_detector_quick.yaml** - Quick training for development
   - Training time: ~2-3 hours
   - Dataset: Small subset
   - Use case: Rapid prototyping, debugging
   - Command: `python scripts/train_edge_detector.py --config docs/training_configs/edge_detector_quick.yaml`

2. **edge_detector_production.yaml** - Full production training
   - Training time: ~24-48 hours
   - Dataset: Full dataset
   - Use case: Final model for deployment
   - Command: `python scripts/train_edge_detector.py --config docs/training_configs/edge_detector_production.yaml`

3. **edge_detector_transfer_learning.yaml** - Fine-tuning pretrained models
   - Training time: ~4-8 hours
   - Dataset: Custom domain data
   - Use case: Adapting to specific image types
   - Command: `python scripts/train_edge_detector.py --config docs/training_configs/edge_detector_transfer_learning.yaml`

### Coefficient Optimizer Models

4. **coefficient_optimizer.yaml** - Regression model training
   - Training time: ~2-4 hours
   - Dataset: Image features and optimal counts
   - Use case: Automatic coefficient count selection
   - Command: `python scripts/train_coefficient_optimizer.py --config docs/training_configs/coefficient_optimizer.yaml`

### Anomaly Detection Models

5. **anomaly_detector.yaml** - Binary classifier training
   - Training time: ~3-6 hours
   - Dataset: Normal and tampered coefficients
   - Use case: Tamper detection before decryption
   - Command: `python scripts/train_anomaly_detector.py --config docs/training_configs/anomaly_detector.yaml`

### Hyperparameter Search

6. **hyperparameter_search.yaml** - Automated hyperparameter optimization
   - Training time: ~24-72 hours
   - Method: Bayesian optimization
   - Use case: Finding optimal hyperparameters
   - Command: `python scripts/hyperparameter_search.py --config docs/training_configs/hyperparameter_search.yaml`

## Configuration Structure

All configuration files follow a consistent structure:

```yaml
training:
  name: model_name
  mode: development|production|transfer_learning|hyperparameter_search
  description: "Model description"

dataset:
  train_path: path/to/train
  val_path: path/to/val
  test_path: path/to/test
  batch_size: 32
  num_workers: 4

model:
  architecture: UNet|MLP|ViT
  # Architecture-specific parameters

optimizer:
  type: Adam|AdamW|SGD
  learning_rate: 1e-4
  weight_decay: 1e-5

scheduler:
  type: ReduceLROnPlateau|CosineAnnealing
  # Scheduler-specific parameters

training_params:
  num_epochs: 100
  early_stopping_patience: 10
  save_frequency: 10

logging:
  use_wandb: true
  wandb_project: project-name
  log_frequency: 50

evaluation:
  metrics: [f1_score, precision, recall]
  targets:
    min_f1_score: 0.90
```

## Customization Guide

### Adjusting for Your Hardware

**Limited GPU Memory:**
```yaml
dataset:
  batch_size: 8  # Reduce from 32
training_params:
  mixed_precision: true  # Enable to save memory
model:
  base_filters: 32  # Reduce from 64
```

**Multiple GPUs:**
```yaml
training_params:
  distributed: true
  num_gpus: 4
  backend: nccl
```

**CPU Only:**
```yaml
training_params:
  device: cpu
  mixed_precision: false
dataset:
  batch_size: 4  # Smaller batches
  num_workers: 0  # Avoid multiprocessing issues
```

### Adjusting for Your Dataset

**Small Dataset (<1000 images):**
```yaml
model:
  dropout: 0.3  # Increase regularization
training_params:
  num_epochs: 50  # Fewer epochs
dataset:
  augmentation: advanced  # More augmentation
```

**Large Dataset (>100k images):**
```yaml
dataset:
  batch_size: 64  # Larger batches
  cache_dataset: true  # Cache in memory
training_params:
  num_epochs: 200
  save_frequency: 20
```

**Imbalanced Dataset:**
```yaml
loss:
  type: BCEWithLogits
  pos_weight: 3.0  # Weight for minority class
training_params:
  class_weights: auto
```

### Adjusting Training Speed

**Faster Training:**
```yaml
model:
  base_filters: 32  # Smaller model
  depth: 3
dataset:
  batch_size: 64  # Larger batches
  num_workers: 8  # More workers
training_params:
  mixed_precision: true
  num_epochs: 50  # Fewer epochs
```

**Better Accuracy (Slower):**
```yaml
model:
  base_filters: 128  # Larger model
  depth: 6
dataset:
  batch_size: 16  # Smaller batches for stability
  augmentation: advanced
training_params:
  num_epochs: 300
  early_stopping_patience: 30
optimizer:
  learning_rate: 5e-5  # Lower LR
```

## Common Training Workflows

### 1. Quick Prototyping

```bash
# Start with quick config
python scripts/train_edge_detector.py --config docs/training_configs/edge_detector_quick.yaml

# Evaluate
python scripts/evaluate_model.py --model models/edge_detector_quick_dev_best.pth --test-data data/test
```

### 2. Hyperparameter Search

```bash
# Run hyperparameter search
python scripts/hyperparameter_search.py --config docs/training_configs/hyperparameter_search.yaml

# Use best config for full training
python scripts/train_edge_detector.py --config config/best_hyperparameters.yaml
```

### 3. Transfer Learning

```bash
# Download pretrained model
wget https://example.com/pretrained_edge_detector.pth -O models/pretrained_edge_detector.pth

# Fine-tune on your data
python scripts/train_edge_detector.py --config docs/training_configs/edge_detector_transfer_learning.yaml
```

### 4. Production Training

```bash
# Full training with all data
python scripts/train_edge_detector.py --config docs/training_configs/edge_detector_production.yaml

# Comprehensive evaluation
python scripts/evaluate_model.py --model models/edge_detector_production_v1_best.pth --test-data data/test --output-dir evaluation_results

# Export for deployment
python scripts/export_model.py --model models/edge_detector_production_v1_best.pth --output models/edge_detector_v1.0.0.pth
```

## Monitoring Training

### Using Weights & Biases

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Enable in config:
```yaml
logging:
  use_wandb: true
  wandb_project: your-project
  wandb_entity: your-team
```
4. View at: https://wandb.ai/your-team/your-project

### Using TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/

# View at http://localhost:6006
```

## Troubleshooting

### Training Not Starting

**Check:**
- Dataset paths are correct
- Required directories exist
- GPU is available (if using CUDA)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

**Solutions:**
- Reduce batch size
- Enable mixed precision
- Reduce model size
- Use gradient accumulation

### Poor Performance

**Check:**
- Data quality and labels
- Learning rate (try LR finder)
- Model capacity
- Training duration

### Slow Training

**Optimize:**
- Increase num_workers
- Enable pin_memory
- Use mixed precision
- Profile code for bottlenecks

## Best Practices

1. **Always start with quick config** for debugging
2. **Use hyperparameter search** before full training
3. **Monitor validation metrics** to detect overfitting
4. **Save checkpoints regularly** to recover from failures
5. **Version your models** with semantic versioning
6. **Document your experiments** in wandb or notebooks
7. **Test on diverse data** before deployment
8. **Compare with baselines** to validate improvements

## Support

For questions or issues:
- Check main documentation: `docs/AI_MODEL_TRAINING.md`
- Review training scripts: `scripts/train_*.py`
- Open an issue on GitHub
- Contact the development team
