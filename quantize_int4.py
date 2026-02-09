"""
INT4 Quantization for ROmniStereo Pruned Model
===============================================

This script applies INT4 quantization (PTQ and QAT) to a pruned ROmniStereo model.

Usage:
------
# PTQ only (fast, no training required)
python quantize_int4.py --restore_ckpt checkpoints/xxx_final.pth --mode ptq

# QAT only (requires training data)
python quantize_int4.py --restore_ckpt checkpoints/xxx_final.pth --mode qat --qat_epochs 3

# Both PTQ and QAT (recommended)
python quantize_int4.py --restore_ckpt checkpoints/xxx_final.pth --mode both --qat_epochs 3

# With custom dataset path
python quantize_int4.py --restore_ckpt checkpoints/xxx_final.pth --db_root /path/to/dataset

Example:
--------
python quantize_int4.py \\
    --restore_ckpt F:/algo/mvs_v119/checkpoints/romnistereo32_v21_bs8_prune_final.pth \\
    --mode both \\
    --qat_epochs 3 \\
    --batch_size 4

Author: Generated for ROmniStereo INT4 Quantization
"""

from __future__ import print_function, division
import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# Import from project
from dataset import Dataset, MultiDataset
from utils.common import *
from utils.image import *
from module.loss_functions import sequence_loss

try:
    from torch.cuda.amp import GradScaler, autocast
except:
    class GradScaler:
        def __init__(self, enabled=False): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass

# ============================================================
# ARGUMENT PARSER
# ============================================================
parser = ArgumentParser(description='INT4 Quantization for ROmniStereo Pruned Model')

# Required
parser.add_argument('--restore_ckpt', required=True, help="Path to pruned checkpoint (from train_prune_v2.py)")

# Dataset
parser.add_argument('--db_root', default='/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/hyp_data_01_trainable/', 
                    type=str, help='Path to dataset')
parser.add_argument('--dbname', nargs='+', default=['omnithings'], type=str, help='Dataset name(s)')

# Quantization mode
parser.add_argument('--mode', choices=['ptq', 'qat', 'both'], default='both',
                    help='Quantization mode: ptq (Post-Training), qat (Quantization-Aware Training), or both')

# QAT training params
parser.add_argument('--qat_epochs', type=int, default=3, help='Number of QAT fine-tuning epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for QAT training')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for QAT')
parser.add_argument('--train_iters', type=int, default=5, help='Number of GRU iterations during training')
parser.add_argument('--valid_iters', type=int, default=5, help='Number of GRU iterations during validation')

# Output
parser.add_argument('--output_dir', default='./checkpoints/quantized', help='Output directory for quantized model')
parser.add_argument('--name', default=None, help='Name for output checkpoint (auto-generated if not provided)')

# Misc
parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
parser.add_argument('--num_bits', type=int, default=4, help='Number of bits for quantization (default: 4)')

args = parser.parse_args()


# ============================================================
# INT4 QUANTIZATION FUNCTIONS
# ============================================================

def quantize_weight_int4(tensor, num_bits=4):
    """
    Per-output-channel INT4 quantization.
    Works for both Conv2d (4D) and Conv3d (5D) tensors.
    
    Args:
        tensor: Weight tensor [out_ch, in_ch, ...] 
        num_bits: Number of bits (default: 4)
    
    Returns:
        Quantized tensor (simulated, still float32)
    """
    # Calculate scale per output channel
    # For Conv2d: [out_ch, in_ch, kH, kW] -> reduce dims [1,2,3]
    # For Conv3d: [out_ch, in_ch, kD, kH, kW] -> reduce dims [1,2,3,4]
    reduce_dims = list(range(1, tensor.dim()))
    scale = tensor.abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.maximum(scale, torch.tensor(1e-8, device=tensor.device, dtype=tensor.dtype))
    
    # Quantize to [-2^(n-1), 2^(n-1)-1] range
    n = 2 ** (num_bits - 1)
    min_val, max_val = -n, n - 1
    
    # Scale -> Round -> Clamp
    scaled = torch.round(tensor / scale * max_val)
    clipped = torch.clamp(scaled, min_val, max_val)
    
    # Dequantize (simulated quantization for inference)
    quantized = clipped * scale / max_val
    
    return quantized


def apply_int4_ptq(model, num_bits=4, verbose=True):
    """
    Apply INT4 Post-Training Quantization to all Conv2d and Conv3d layers.
    
    Args:
        model: PyTorch model
        num_bits: Number of bits for quantization
        verbose: Print layer names
    
    Returns:
        Quantized model (in-place modification)
    """
    quantized_count = 0
    skipped_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            if hasattr(module, 'weight') and module.weight is not None:
                try:
                    original_weight = module.weight.data.clone()
                    quantized_weight = quantize_weight_int4(original_weight, num_bits)
                    
                    # Check for invalid values
                    if torch.isnan(quantized_weight).any() or torch.isinf(quantized_weight).any():
                        raise ValueError(f"Invalid values in quantized weights for {name}")
                    
                    module.weight.data.copy_(quantized_weight)
                    quantized_count += 1
                    
                    if verbose:
                        layer_type = "Conv2d" if isinstance(module, nn.Conv2d) else "Conv3d"
                        print(f"  Quantized [{layer_type}]: {name}")
                        
                except Exception as e:
                    print(f"  Skipped {name}: {str(e)}")
                    skipped_count += 1
    
    print(f"\nQuantization Summary:")
    print(f"  Quantized layers: {quantized_count}")
    print(f"  Skipped layers: {skipped_count}")
    
    return model


# ============================================================
# FAKE QUANTIZE FOR QAT (Quantization-Aware Training)
# ============================================================

class FakeQuantize(torch.autograd.Function):
    """
    Fake quantization function for QAT.
    Forward: Quantize weights
    Backward: Straight-Through Estimator (gradient passes through)
    """
    @staticmethod
    def forward(ctx, x, num_bits=4):
        ctx.num_bits = num_bits
        ctx.save_for_backward(x)
        
        # Per-channel scale
        reduce_dims = list(range(1, x.dim()))
        scale = x.abs().amax(dim=reduce_dims, keepdim=True)
        scale = torch.maximum(scale, torch.tensor(1e-8, device=x.device, dtype=x.dtype))
        
        n = 2 ** (num_bits - 1)
        min_val, max_val = -n, n - 1
        
        x_scaled = torch.round(x / scale * max_val)
        x_clipped = torch.clamp(x_scaled, min_val, max_val)
        x_quantized = x_clipped * scale / max_val
        
        return x_quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: gradient passes through unchanged
        return grad_output, None


class QuantizedConv2d(nn.Module):
    """Wrapper for Conv2d with fake quantization during QAT training."""
    def __init__(self, conv_module, num_bits=4):
        super().__init__()
        self.conv = conv_module
        self.num_bits = num_bits
    
    def forward(self, x):
        # Apply fake quantization to weights
        quantized_weight = FakeQuantize.apply(self.conv.weight, self.num_bits)
        
        return F.conv2d(
            x, quantized_weight, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups
        )


class QuantizedConv3d(nn.Module):
    """Wrapper for Conv3d with fake quantization during QAT training."""
    def __init__(self, conv_module, num_bits=4):
        super().__init__()
        self.conv = conv_module
        self.num_bits = num_bits
    
    def forward(self, x):
        # Apply fake quantization to weights
        quantized_weight = FakeQuantize.apply(self.conv.weight, self.num_bits)
        
        return F.conv3d(
            x, quantized_weight, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups
        )


def apply_qat_wrappers(model, num_bits=4):
    """
    Replace Conv2d/Conv3d layers with QAT wrappers.
    This enables fake quantization during training.
    
    Args:
        model: PyTorch model
        num_bits: Number of bits for quantization
    
    Returns:
        Model with QAT wrappers
    """
    wrapped_count = 0
    
    def replace_conv(module, name, parent):
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                setattr(module, child_name, QuantizedConv2d(child, num_bits))
                nonlocal wrapped_count
                wrapped_count += 1
            elif isinstance(child, nn.Conv3d):
                setattr(module, child_name, QuantizedConv3d(child, num_bits))
                wrapped_count += 1
            else:
                replace_conv(child, child_name, module)
    
    replace_conv(model, '', None)
    print(f"Wrapped {wrapped_count} layers with QAT fake quantization")
    
    return model


def remove_qat_wrappers(model):
    """
    Remove QAT wrappers and apply final quantization.
    Call this after QAT training to get the final quantized model.
    """
    def unwrap(module):
        for child_name, child in module.named_children():
            if isinstance(child, (QuantizedConv2d, QuantizedConv3d)):
                # Get the inner conv and quantize its weights
                inner_conv = child.conv
                quantized_weight = quantize_weight_int4(inner_conv.weight.data, child.num_bits)
                inner_conv.weight.data.copy_(quantized_weight)
                setattr(module, child_name, inner_conv)
            else:
                unwrap(child)
    
    unwrap(model)
    return model


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, dataset, grids, valid_iters=5):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        dataset: Dataset object
        grids: Lookup table grids
        valid_iters: Number of GRU iterations
    
    Returns:
        mean_errors: [>1, >3, >5, MAE, RMS]
    """
    model.eval()
    eval_list = dataset.opts.test_idx
    errors = np.zeros((len(eval_list), 5))
    
    with torch.no_grad():
        for d, fidx in enumerate(tqdm(eval_list, desc="Evaluating", leave=False)):
            imgs, gt, valid, _ = dataset.loadSample(fidx)
            imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]
            
            invdepth_idx = model(imgs, grids, valid_iters, test_mode=True)
            invdepth_idx = toNumpy(invdepth_idx[0, 0])
            
            errors[d, :] = dataset.evalError(invdepth_idx, gt, valid)
    
    mean_errors = errors.mean(axis=0)
    return mean_errors


def print_metrics(errors, label=""):
    """Print evaluation metrics."""
    print(f"{label}")
    print(f"  >1: {errors[0]:.4f}%")
    print(f"  >3: {errors[1]:.4f}%")
    print(f"  >5: {errors[2]:.4f}%")
    print(f"  MAE: {errors[3]:.4f}")
    print(f"  RMS: {errors[4]:.4f}")


# ============================================================
# QAT TRAINING LOOP
# ============================================================

def train_qat(model, dataloader, grids, dataset, epochs, lr, train_iters, valid_iters, mixed_precision=False):
    """
    QAT fine-tuning loop.
    
    Args:
        model: Model with QAT wrappers
        dataloader: Training data loader
        grids: Lookup table grids
        dataset: Dataset for validation
        epochs: Number of training epochs
        lr: Learning rate
        train_iters: GRU iterations during training
        valid_iters: GRU iterations during validation
        mixed_precision: Use mixed precision training
    
    Returns:
        best_model: Best model based on RMS
        best_rms: Best RMS achieved
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=mixed_precision)
    
    best_rms = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"QAT Epoch {epoch+1}/{epochs}")
        
        for imgs_b, gt_b, valid_b, _ in pbar:
            imgs_b = [img.cuda() for img in imgs_b]
            gt_b, valid_b = gt_b.cuda(), valid_b.cuda()
            
            optimizer.zero_grad()
            
            predictions = model(imgs_b, grids, train_iters)
            loss = sequence_loss(predictions, gt_b.unsqueeze(1), valid_b.unsqueeze(1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Validation
        avg_loss = total_loss / num_batches
        metrics = evaluate(model, dataset, grids, valid_iters)
        current_rms = metrics[4]
        
        print(f"\nEpoch {epoch+1}/{epochs}: Avg Loss={avg_loss:.4f}, RMS={current_rms:.4f}")
        
        # Save best model
        if current_rms < best_rms:
            best_rms = current_rms
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  New best model! RMS={best_rms:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_rms


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in MB (assuming float32)."""
    param_size = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
    return param_size / (1024 * 1024)


def get_quantized_size_mb(model, num_bits=4):
    """Estimate quantized model size in MB."""
    param_size = sum(p.numel() * num_bits / 8 for p in model.parameters())
    return param_size / (1024 * 1024)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("INT4 Quantization for ROmniStereo")
    print("=" * 60)
    
    # ========================
    # 1. Load Pruned Checkpoint
    # ========================
    print(f"\n[1] Loading pruned checkpoint: {args.restore_ckpt}")
    
    if not os.path.exists(args.restore_ckpt):
        sys.exit(f"ERROR: Checkpoint not found: {args.restore_ckpt}")
    
    checkpoint = torch.load(args.restore_ckpt, weights_only=False)
    
    # Load model object (saved by train_prune_v2.py)
    if 'model' in checkpoint:
        model = checkpoint['model']
        print("  Loaded full model object from checkpoint")
    else:
        sys.exit("ERROR: Checkpoint does not contain 'model' key. Use train_prune_v2.py format.")
    
    model = model.cuda()
    
    # Get data_opts from checkpoint
    if 'data_opts' in checkpoint:
        data_opts = checkpoint['data_opts']
    else:
        # Fallback to default
        data_opts = Edict({
            'phi_deg': 45.0,
            'num_invdepth': 48,
            'equirect_size': [128, 400],
            'num_downsample': 1,
            'use_rgb': False
        })
    
    # Print model info
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    quantized_size = get_quantized_size_mb(model, args.num_bits)
    
    print(f"  Parameters: {num_params:,}")
    print(f"  Model size (FP32): {model_size:.2f} MB")
    print(f"  Estimated INT{args.num_bits} size: {quantized_size:.2f} MB")
    print(f"  Compression ratio: {model_size/quantized_size:.1f}x")
    
    # ========================
    # 2. Load Dataset
    # ========================
    print(f"\n[2] Loading dataset: {args.dbname}")
    
    if len(args.dbname) > 1:
        dataset = MultiDataset(args.dbname, data_opts, db_root=args.db_root)
    else:
        dataset = Dataset(args.dbname[0], data_opts, db_root=args.db_root)
    
    grids = [torch.tensor(grid).cuda() for grid in dataset.grids]
    
    print(f"  Train samples: {len(dataset.train_idx)}")
    print(f"  Test samples: {len(dataset.test_idx)}")
    
    # ========================
    # 3. Baseline Evaluation
    # ========================
    print(f"\n[3] Baseline Evaluation (FP32 Pruned)")
    baseline_metrics = evaluate(model, dataset, grids, args.valid_iters)
    baseline_rms = baseline_metrics[4]
    print_metrics(baseline_metrics, "  Baseline Metrics:")
    
    # ========================
    # 4. Apply Quantization
    # ========================
    results = {
        'baseline_rms': baseline_rms,
        'ptq_rms': None,
        'qat_rms': None,
    }
    
    if args.mode in ['ptq', 'both']:
        print(f"\n[4] Applying INT{args.num_bits} PTQ (Post-Training Quantization)")
        
        # Create a copy for PTQ
        ptq_model = copy.deepcopy(model)
        ptq_model = apply_int4_ptq(ptq_model, args.num_bits, verbose=True)
        
        # Evaluate PTQ
        print("\n  Evaluating PTQ model...")
        ptq_metrics = evaluate(ptq_model, dataset, grids, args.valid_iters)
        ptq_rms = ptq_metrics[4]
        results['ptq_rms'] = ptq_rms
        
        print_metrics(ptq_metrics, "  PTQ Metrics:")
        print(f"  Delta vs Baseline: {ptq_rms - baseline_rms:+.4f}")
    
    if args.mode in ['qat', 'both']:
        print(f"\n[5] Applying INT{args.num_bits} QAT (Quantization-Aware Training)")
        
        # Create dataloader for QAT
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=0, 
            drop_last=True
        )
        
        # Start from PTQ model if available, otherwise from baseline
        if args.mode == 'both':
            qat_model = copy.deepcopy(ptq_model)
            print("  Starting QAT from PTQ model")
        else:
            qat_model = copy.deepcopy(model)
            # Apply PTQ first
            qat_model = apply_int4_ptq(qat_model, args.num_bits, verbose=False)
            print("  Applied PTQ before QAT")
        
        # Apply QAT wrappers
        qat_model = apply_qat_wrappers(qat_model, args.num_bits)
        
        # Train with QAT
        print(f"\n  Starting QAT training for {args.qat_epochs} epochs...")
        qat_model, best_rms = train_qat(
            qat_model, dataloader, grids, dataset,
            epochs=args.qat_epochs,
            lr=args.lr,
            train_iters=args.train_iters,
            valid_iters=args.valid_iters,
            mixed_precision=args.mixed_precision
        )
        
        # Remove QAT wrappers and finalize quantization
        print("\n  Finalizing quantized model...")
        qat_model = remove_qat_wrappers(qat_model)
        
        # Final evaluation
        print("  Final QAT evaluation...")
        qat_metrics = evaluate(qat_model, dataset, grids, args.valid_iters)
        qat_rms = qat_metrics[4]
        results['qat_rms'] = qat_rms
        
        print_metrics(qat_metrics, "  QAT Final Metrics:")
        print(f"  Delta vs Baseline: {qat_rms - baseline_rms:+.4f}")
    
    # ========================
    # 5. Save Quantized Model
    # ========================
    print(f"\n[6] Saving Quantized Model")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which model to save
    if args.mode == 'qat' or args.mode == 'both':
        final_model = qat_model
        mode_suffix = 'qat'
    else:
        final_model = ptq_model
        mode_suffix = 'ptq'
    
    # Generate output filename
    if args.name:
        output_name = args.name
    else:
        base_name = os.path.splitext(os.path.basename(args.restore_ckpt))[0]
        output_name = f"{base_name}_int{args.num_bits}_{mode_suffix}"
    
    output_path = os.path.join(args.output_dir, f"{output_name}.pth")
    
    # Save checkpoint
    save_dict = {
        'model': final_model,
        'net_state_dict': final_model.state_dict(),
        'net_opts': checkpoint.get('net_opts', None),
        'data_opts': data_opts,
        'quantization': {
            'method': f'int{args.num_bits}',
            'mode': args.mode,
            'num_bits': args.num_bits,
            'qat_epochs': args.qat_epochs if args.mode in ['qat', 'both'] else 0,
        },
        'metrics': results,
        'original_ckpt': args.restore_ckpt,
    }
    
    torch.save(save_dict, output_path)
    print(f"  Saved to: {output_path}")
    
    # ========================
    # 6. Summary
    # ========================
    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    print(f"\n{'Method':<20} {'RMS':<12} {'Delta':>10}")
    print("-" * 42)
    print(f"{'Baseline (FP32)':<20} {baseline_rms:<12.4f} {'-':>10}")
    
    if results['ptq_rms'] is not None:
        delta = results['ptq_rms'] - baseline_rms
        print(f"{'INT4 PTQ':<20} {results['ptq_rms']:<12.4f} {delta:>+10.4f}")
    
    if results['qat_rms'] is not None:
        delta = results['qat_rms'] - baseline_rms
        print(f"{'INT4 QAT':<20} {results['qat_rms']:<12.4f} {delta:>+10.4f}")
    
    print("-" * 42)
    print(f"\nModel Size:")
    print(f"  FP32: {model_size:.2f} MB")
    print(f"  INT{args.num_bits}: {quantized_size:.2f} MB (estimated)")
    print(f"  Compression: {model_size/quantized_size:.1f}x")
    
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
