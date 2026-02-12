# PyTorch Integration Guide

This document describes the PyTorch upgrade and how to use PyTorch neural network models in the Marketing Performance Predictor.

## Overview

The system now supports PyTorch neural network models alongside the existing scikit-learn models (Linear, Tree, Forest, SVR). PyTorch models provide deep learning capabilities with feedforward and deep network architectures.

## What's New

### 1. PyTorch Model Classes (`model/pytorch_model.py`)

- **FeedforwardNet**: Standard feedforward neural network with configurable hidden layers
- **DeepNet**: Deep neural network with residual connections
- **PyTorchRegressor**: Wrapper class compatible with scikit-learn interface

### 2. Features

- **Automatic Architecture Search**: Tests multiple hidden layer configurations
- **Early Stopping**: Prevents overfitting with configurable patience
- **Batch Normalization**: Improves training stability
- **Dropout**: Regularization to prevent overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **GPU Support**: Automatically uses CUDA if available

### 3. Integration Points

- **Training Pipeline**: Added `pytorch()` method to `Regression` class
- **Model Evaluation**: PyTorch models evaluated alongside other models
- **Model Saving**: PyTorch models saved using joblib (includes model state and scaler)
- **API Support**: API automatically handles PyTorch model predictions

## Usage

### Training with PyTorch Models

To train models including PyTorch:

```python
from training import train

# Train with PyTorch models
train(
    output='impressions',
    models=['linear', 'forest', 'pytorch'],
    print_output=True
)
```

### Training Only PyTorch Models

```python
train(
    output='impressions',
    models=['pytorch'],
    print_output=True
)
```

### Model Types

The PyTorch implementation supports two architectures:

1. **Feedforward** (default): Standard multi-layer perceptron
   - Configurable hidden layers: [128, 64, 32] (default)
   - Dropout rate: 0.2 (default)

2. **Deep**: Deep network with residual connections
   - Configurable hidden layers: [256, 128, 64, 32] (default)
   - Dropout rate: 0.3 (default)

### Customizing PyTorch Models

You can customize PyTorch models in `regression.py`:

```python
def pytorch(self, model_type='feedforward'):
    # model_type can be 'feedforward' or 'deep'
    # Hidden sizes and dropout can be customized
    regressor = PyTorchRegressor(
        model_type=model_type,
        hidden_sizes=[256, 128, 64],  # Custom architecture
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        early_stopping_patience=10
    )
```

## Model Architecture Details

### FeedforwardNet
- Input layer → Hidden layers (ReLU + BatchNorm + Dropout) → Output layer
- Configurable number and size of hidden layers
- Dropout for regularization

### DeepNet
- Input layer → Multiple hidden blocks with residual connections
- Batch normalization and dropout in each block
- Residual connections when dimensions match

## Training Parameters

- **Learning Rate**: 0.001 (default)
- **Batch Size**: 32 (default)
- **Epochs**: 100 (default)
- **Early Stopping Patience**: 10 epochs (default)
- **Optimizer**: Adam with weight decay (1e-5)
- **Loss Function**: MSE (Mean Squared Error)
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

## Dependencies

### New Dependencies Added

- `torch==2.1.2` - PyTorch core library
- `torchvision==0.16.2` - Vision utilities (for future extensions)
- `torchaudio==2.1.2` - Audio utilities (for future extensions)

### Updated Files

- `model/pyproject.toml` - Added PyTorch dependencies
- `api/pyproject.toml` - Added PyTorch for inference
- `api/requirements.txt` - Added PyTorch
- `Dockerfile.model` - Added PyTorch installation
- `Dockerfile.api` - Added PyTorch installation

## Performance Considerations

### GPU Acceleration

PyTorch models automatically use GPU if available:
- CUDA devices are automatically detected
- Falls back to CPU if GPU is not available
- No code changes needed

### Memory Usage

- PyTorch models may use more memory than scikit-learn models
- Batch size can be adjusted to fit available memory
- Models are saved in evaluation mode to reduce memory footprint

### Training Time

- PyTorch models typically take longer to train than scikit-learn models
- Early stopping helps reduce unnecessary training time
- Cross-validation is performed for model selection

## Model Comparison

The system evaluates all models (including PyTorch) and selects the best performing one based on test accuracy:

```python
# Example output during training
Linear score: 0.85
Forest score: 0.87
PyTorch feedforward score (hidden=[128, 64, 32]): 0.89
PyTorch feedforward score (hidden=[256, 128, 64]): 0.88
Best PyTorch feedforward params: {'hidden_sizes': [128, 64, 32], 'dropout_rate': 0.2}
Best PyTorch feedforward score: 0.89
Regressors evaluated. Best regressor is: PyTorchRegressor(...)
```

## API Usage

The API automatically handles PyTorch models:

```python
# No changes needed in API calls
POST /campaign
{
    "cost": 10000,
    "start_month": 1,
    ...
}

# API automatically:
# 1. Loads model (handles both joblib and PyTorch)
# 2. Prepares input data
# 3. Makes prediction (handles PyTorch tensor conversion)
# 4. Returns result
```

## Troubleshooting

### Model Loading Issues

If you encounter issues loading PyTorch models:

1. Ensure PyTorch is installed: `pip install torch`
2. Check model file exists: `./models/impressions_model.pkl`
3. Verify model was saved correctly during training

### GPU Issues

If GPU is not being used:

1. Check CUDA availability: `torch.cuda.is_available()`
2. Verify CUDA installation
3. Models will automatically fall back to CPU

### Memory Errors

If you encounter out-of-memory errors:

1. Reduce batch size: `batch_size=16` or `batch_size=8`
2. Reduce model size: Smaller `hidden_sizes`
3. Use CPU instead of GPU if GPU memory is limited

## Future Enhancements

Potential improvements:

- [ ] Transformer-based models for sequential data
- [ ] Attention mechanisms
- [ ] Ensemble of PyTorch models
- [ ] Hyperparameter optimization with Optuna
- [ ] Model quantization for faster inference
- [ ] ONNX export for deployment optimization

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Neural Network Best Practices](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)

