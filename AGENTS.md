ROmniStereo Project

## Overview
Depth estimation model using omnidirectional cameras.
Target: Deploy on Jetson Orin Nano with INT4/INT8 quantization.

## Key Files

### Training & Pruning
- `train_u.py` - Train model from scratch
- `train_prune_v2.py` - Prune model (saves full model object)

### Quantization
- `quantize_int4.py` - Simple INT4 quantization (PTQ + QAT)
- `quantize_advanced.py` - Advanced multi-method quantization
- `quantization/` - Quantization module:
  - `utils.py` - Core functions (scale/zp, fake_quantize)
  - `quantizers.py` - MinMax, Percentile, MSE quantizers
  - `calibration.py` - Calibration data collection
  - `adaround.py` - AdaRound algorithm
  - `gptq.py` - GPTQ algorithm
  - `smoothquant.py` - SmoothQuant (W+A quantization)

### Export & Inference
- `export_onnx_jetson_v2.py` - Export to ONNX
- `inference.py` - Inference with ONNX

### Model Architecture
- `module/network.py` - ROmniStereo model definition
- `module/featurelayer.py` - Conv2D encoder (18 layers)
- `module/update.py` - ConvGRU, DepthHead, MotionEncoder

### Data
- `dataset.py` - Dataset class for OmniThings/Omnihouse

## Model Architecture
- **FeatureLayers**: Conv2D layers (encoder)
- **Generator**: Conv3D layers (volume)
- **UpdateBlock**: ConvGRU, MotionEncoder, DepthHead
- **Important**: 100% Conv2D/Conv3D - NO nn.Linear layers

## Checkpoint Format (train_prune_v2.py)
```python
{
    'model': model,           # Full pruned model object
    'net_state_dict': ...,    # State dict
    'net_opts': {...},        # Network options
    'data_opts': {...},       # Data options (phi_deg, num_invdepth, etc.)
    'original_params': int,
    'final_params': int,
}
```

## Pipeline
```
train_u.py → train_prune_v2.py → quantize_int4.py → export_onnx_jetson_v2.py → inference.py
```

