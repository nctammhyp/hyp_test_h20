# Session Summary - ROmniStereo INT4 Quantization Project

**Date**: 2026-02-07  
**Project Path**: `F:\algo\mvs_v119`  
**Model**: ROmniStereo - Multi-view stereo depth estimation

---

## Project Overview

ROmniStereo is a depth estimation model using omnidirectional cameras. The model architecture:
- **FeatureLayers** (Encoder): 18 Conv2D layers
- **Generator** (Volume): Conv3D layers (MLP blocks)
- **UpdateBlock**: ConvGRU, MotionEncoder, DepthHead - all Conv2D
- **Important**: Model has **NO nn.Linear layers** - 100% Conv2D/Conv3D

---

## What We Did

### 1. Analyzed Pruning Checkpoint Loading Issue

**Problem**: Original `train_prune.py` only saves `state_dict`, not the full model object. When model is pruned with `torch_pruning`, the structure changes (channels reduced). Cannot load pruned `state_dict` into a fresh `ROmniStereo()` model - size mismatch error.

**Existing pruned checkpoint**: `F:\algo\mvs_v119\checkpoints\romnistereo32_v20_bs8_prune_step9_final.pth`

### 2. Created Two New Files

#### File 1: `train_prune_v2.py`
- Modified version of `train_prune.py`
- Key change: Saves full model object in checkpoint (`'model': model`)
- Also saves `data_opts`, `original_params`, `final_params`, `reduction_percent`
- Adds parameter counting and logging

#### File 2: `export_onnx_jetson_v2.py`
- Modified version of `export_onnx_jetson.py`
- Key change: Loads `checkpoint['model']` directly instead of creating new model
- Has fallback for old checkpoint format
- Verifies ONNX after export

### 3. Researched INT4 Quantization for Jetson Deployment

**User Requirements**:
- Quantize pruned model to INT4
- Deploy on Jetson Orin Nano
- Use TensorRT for inference

**Libraries Researched**:

| Library | Conv2D Support | INT4 | TensorRT | Recommendation |
|---------|---------------|------|----------|----------------|
| torchao | Linear only | Yes | Limited | Not suitable for this model |
| NVIDIA ModelOpt | Yes | Yes | Best | Recommended for Jetson |
| pytorch-quantization | Yes | Yes | Yes | Legacy |
| PyTorch native | Yes | INT8 only | Limited | Backup |

### 4. Analyzed User's Existing Notebook

**File**: `F:\algo\mvs_v119\quantize\test-2.ipynb`

This notebook demonstrates **custom INT4 quantization for VGG (Conv2D model)** on CIFAR-100:

**Key Approach in Notebook**:
```python
# Custom INT4 quantization function (NOT torchao)
def quantize_x_(tensor, config):
    num_bits = 4
    scale = tensor.abs().amax(dim=list(range(1, tensor.dim())), keepdim=True)
    scale = torch.maximum(scale, torch.tensor(1e-8))
    min_val = -(2**(num_bits-1))  # -8
    max_val = 2**(num_bits-1) - 1  # 7
    scaled_tensor = torch.clamp(torch.round(tensor / scale * max_val), min_val, max_val)
    quantized = scaled_tensor * scale / max_val
    return quantized

# Apply to both Conv2D and Linear
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        quantized_weight = quantize_x_(module.weight.data, ...)
        module.weight.data.copy_(quantized_weight)
```

**Notebook Results**:
- Original FP32: 70.77% accuracy, 39.2 MB
- INT4 PTQ: 69.07% accuracy, 4.9 MB
- INT4 + QAT (3 epochs): 69.69% accuracy, 4.9 MB
- torchao `int8_weight_only()`: Works with Conv2D!

---

## Key Files in Project

| File | Purpose |
|------|---------|
| `train_prune.py` | Original pruning script (old format) |
| `train_prune_v2.py` | **NEW** - Saves full model object |
| `export_onnx_jetson.py` | Original ONNX export |
| `export_onnx_jetson_v2.py` | **NEW** - Loads pruned model correctly |
| `train_u.py` | Original training script |
| `module/network.py` | ROmniStereo model definition |
| `module/featurelayer.py` | Conv2D encoder (18 layers) |
| `module/update.py` | ConvGRU, DepthHead, MotionEncoder |
| `quantize/test-2.ipynb` | Example INT4 quantization for Conv2D |

---

## Model Structure Details

### FeatureLayers (module/featurelayer.py)
```
- Conv2D(in_channel, CH, 5, stride=2)  # conv[1] - downsample
- Conv2D(CH, CH, 3) x 10              # conv[2-11] - with residual connections
- Conv2D(CH, CH, 3, dilation=d) x 6   # conv[12-17] - dilated convolutions
- Conv2D(CH, CH, 3, bn=False)         # conv[18] - output
```

### UpdateBlock (module/update.py)
```
- MotionEncoder: 5 Conv2D layers
- ConvGRU: 3 Conv2D layers
- DepthHead: 2 Conv2D layers
- Mask: 2 Conv2D layers
```

---

## Next Steps to Implement

### 1. Create `quantize_int4_custom.py`
Custom INT4 quantization for Conv2D/Conv3D based on notebook approach:
- PTQ (Post-Training Quantization)
- QAT (Quantization-Aware Training) with FakeQuantize

### 2. Create `quantize_int8_torchao.py`
Using torchao INT8 as backup option:
- `int8_weight_only()` confirmed working with Conv2D
- 2x compression (vs 4x for INT4)

### 3. (Optional) Create `quantize_modelopt.py`
Using NVIDIA ModelOpt for TensorRT:
- Best for Jetson deployment
- Inserts QDQ nodes in ONNX
- Requires JetPack 6.x for INT4 support

### 4. Create `export_onnx_quantized.py`
Export quantized model with QDQ nodes for TensorRT

### 5. Create `inference_quantized.py`
Test and benchmark quantized models

---

## Hardware Target

- **Jetson Orin Nano** (4GB or 8GB - not confirmed)
- **JetPack version**: Not confirmed (need 6.x for INT4 TensorRT support)
- **TensorRT**: Required for final deployment

---

## Pending Questions

1. JetPack version on Jetson?
2. Orin Nano 4GB or 8GB?
3. Which INT4 approach to implement:
   - Custom INT4 (like notebook)
   - NVIDIA ModelOpt
   - Or both for comparison

---

## How to Continue Next Session

Copy and paste this prompt to start a new conversation:

```
Continue working on F:\algo\mvs_v119 project. 

Read the file SESSION_SUMMARY.md for context about what we did.

We're implementing INT4 quantization for ROmniStereo model (Conv2D-based depth estimation) for Jetson Orin Nano deployment.

Key files created:
- train_prune_v2.py (saves full model object)
- export_onnx_jetson_v2.py (loads pruned model correctly)

Reference: quantize/test-2.ipynb shows custom INT4 approach for Conv2D that works.

Next: Create quantization scripts using custom INT4 approach from notebook, adapted for ROmniStereo's Conv2D/Conv3D layers. Target is TensorRT deployment on Jetson.
```

---

## Notes

- Model is 100% convolutional - no nn.Linear layers
- torchao INT4 does NOT work (only supports Linear layers)
- Custom INT4 from notebook DOES work with Conv2D
- Must save full model object when pruning (use train_prune_v2.py)
