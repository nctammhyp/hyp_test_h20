import torch
import torch.nn as nn
import argparse
import os
from easydict import EasyDict as Edict
import onnx

# Import model definition (chi can cho fallback)
from module.network import ROmniStereo

# ==========================================
# 1. CONFIGURATION (Fallback cho checkpoint cu)
# ==========================================
def get_default_opts():
    """Default options - chi dung khi load checkpoint KHONG co model object"""
    opts = Edict()
    opts.data_opts = Edict({
        'phi_deg': 45.0, 
        'num_invdepth': 48, 
        'equirect_size': [128, 400], 
        'num_downsample': 1, 
        'use_rgb': False
    })
    opts.net_opts = Edict({
        'base_channel': 8,
        'num_invdepth': 48, 
        'use_rgb': False, 
        'encoder_downsample_twice': False, 
        'num_downsample': 1, 
        'corr_levels': 4, 
        'corr_radius': 4, 
        'mixed_precision': False, 
        'fix_bn': False
    })
    return opts

# ==========================================
# 2. MODEL WRAPPER FOR ONNX
# ==========================================
class ROmniStereoONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img0, img1, img2, grid0, grid1, grid2):
        # Gom input thanh list nhu model yeu cau
        imgs = [img0, img1, img2]
        grids = [grid0, grid1, grid2]
        
        # Quan trong: test_mode=True de lay ket qua depth cuoi cung
        # iters=5 la so vong lap GRU mac dinh luc inference
        return self.model(imgs, grids, iters=5, test_mode=True)

# ==========================================
# 3. MAIN EXPORT FUNCTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Export Pruned Model to ONNX for Jetson')
    
    # Duong dan den file .pth da train voi train_prune_v2.py
    parser.add_argument('--ckpt_path', type=str, 
                        default=r"F:\algo\mvs_v119\checkpoints\romnistereo32_v21_bs8_prune_final_int4_qat.pth", 
                        help="Path to .pth checkpoint (from train_prune_v2.py)")
    parser.add_argument('--output_path', type=str, 
                        default=r"checkpoints/onnx/romnistereo32_v21_bs8_prune_final_int4_qat.onnx", 
                        help="Output ONNX file path")
    parser.add_argument('--opset', type=int, default=11, 
                        help="ONNX opset version (11 for TensorRT compatibility)")
    args = parser.parse_args()

    # ==========================================
    # 1. LOAD MODEL
    # ==========================================
    print(f"Loading checkpoint: {args.ckpt_path}")
    
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    
    checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    
    # ==========================================
    # KIEM TRA VA LOAD MODEL THEO CACH PHU HOP
    # ==========================================
    if 'model' in checkpoint:
        # ==========================================
        # TRUONG HOP 1: Checkpoint tu train_prune_v2.py (CO model object)
        # ==========================================
        model = checkpoint['model']
        print("Loaded PRUNED model object directly from checkpoint!")
        
        # In thong tin ve model da prune
        if 'original_params' in checkpoint and 'final_params' in checkpoint:
            print(f"  - Original params: {checkpoint['original_params']:,}")
            print(f"  - Final params: {checkpoint['final_params']:,}")
            print(f"  - Reduction: {checkpoint.get('reduction_percent', 'N/A'):.1f}%")
        elif 'current_params' in checkpoint:
            print(f"  - Current params: {checkpoint['current_params']:,}")
            
        if 'rms' in checkpoint:
            print(f"  - Best RMS: {checkpoint['rms']:.4f}")
            
        # Lay data_opts tu checkpoint neu co
        if 'data_opts' in checkpoint:
            data_opts = checkpoint['data_opts']
        else:
            data_opts = get_default_opts().data_opts
            print("  - Warning: data_opts not in checkpoint, using defaults")
            
    else:
        # ==========================================
        # TRUONG HOP 2: Checkpoint cu (CHI CO state_dict) - KHONG HOAT DONG VOI PRUNED MODEL
        # ==========================================
        print("WARNING: Checkpoint does not contain 'model' object!")
        print("         This export method only works with train_prune_v2.py checkpoints.")
        print("         Attempting fallback load (may fail for pruned models)...")
        
        opts = get_default_opts()
        data_opts = opts.data_opts
        
        # Su dung net_opts tu checkpoint neu co
        if 'net_opts' in checkpoint:
            net_opts = checkpoint['net_opts']
        else:
            net_opts = opts.net_opts
            
        model = ROmniStereo(net_opts)
        
        # Load state dict
        state_dict = checkpoint.get('net_state_dict', checkpoint)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("Loaded state_dict successfully (non-pruned model)")
        except RuntimeError as e:
            print(f"ERROR: Cannot load state_dict - size mismatch!")
            print(f"       This checkpoint appears to be from a PRUNED model,")
            print(f"       but was saved WITHOUT the model object.")
            print(f"       Please re-train using train_prune_v2.py to create a compatible checkpoint.")
            raise e

    model.eval()
    model.cpu()  # Chuyen ve CPU de export
    
    # ==========================================
    # 2. WRAP MODEL CHO ONNX
    # ==========================================
    onnx_wrapper = ROmniStereoONNX(model)

    # ==========================================
    # 3. CHUAN BI DUMMY INPUTS
    # ==========================================
    print("Preparing dummy inputs...")
    
    # Kich thuoc input image
    H_in, W_in = 384, 400
    C_in = 1  # Grayscale
    
    # Images
    img0 = torch.randn(1, C_in, H_in, W_in)
    img1 = torch.randn(1, C_in, H_in, W_in)
    img2 = torch.randn(1, C_in, H_in, W_in)

    # Tinh toan shape grid dua tren config
    num_downsample = data_opts.get('num_downsample', 1)
    equirect_size = data_opts.get('equirect_size', [128, 400])
    num_invdepth = data_opts.get('num_invdepth', 48)
    
    H_out = equirect_size[0] // (2 ** num_downsample)
    W_out = equirect_size[1] // (2 ** num_downsample)
    D_out = num_invdepth // (2 ** num_downsample)
    
    grid_shape = (H_out, W_out, D_out, 2)
    print(f"Using dummy grid shape: {grid_shape}")
    
    grid0 = torch.randn(*grid_shape)
    grid1 = torch.randn(*grid_shape)
    grid2 = torch.randn(*grid_shape)

    # ==========================================
    # 4. EXPORT TO ONNX
    # ==========================================
    # Dam bao thu muc output ton tai
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"Exporting to {args.output_path} (Opset {args.opset})...")
    
    torch.onnx.export(
        onnx_wrapper,
        (img0, img1, img2, grid0, grid1, grid2),
        args.output_path,
        input_names=["img0", "img1", "img2", "grid0", "grid1", "grid2"],
        output_names=["depth_map"],
        opset_version=args.opset,
        do_constant_folding=True,
        keep_initializers_as_inputs=False
    )
    
    # ==========================================
    # 5. VERIFY ONNX MODEL
    # ==========================================
    print("Verifying ONNX model...")
    onnx_model = onnx.load(args.output_path)
    onnx.checker.check_model(onnx_model)
    
    # In thong tin model
    file_size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"\nExport successful!")
    print(f"  - Output file: {args.output_path}")
    print(f"  - File size: {file_size_mb:.2f} MB")
    print(f"  - Opset version: {args.opset}")
    print(f"\nCopy this file to your Jetson Orin Nano for TensorRT conversion.")

if __name__ == "__main__":
    main()
