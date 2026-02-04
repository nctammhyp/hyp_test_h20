import torch
import torch.nn as nn
import argparse
import os
from easydict import EasyDict as Edict
import onnx

# Import model definition
from module.network import ROmniStereo

# ==========================================
# 1. CONFIGURATION (Theo yêu cầu của bạn)
# ==========================================
def get_opts():
    opts = Edict()
    # Config khớp với export_onnx cũ
    opts.data_opts = Edict({
        'phi_deg': 45.0, 
        'num_invdepth': 128, 
        'equirect_size': [128, 400], 
        'num_downsample': 1, 
        'use_rgb': False
    })
    opts.net_opts = Edict({
        'base_channel': 16, # Lưu ý: Bạn dùng 16, code cũ dùng 32. Hãy chắc chắn model .pth được train với 16.
        'num_invdepth': 128, 
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
        # Gom input thành list như model yêu cầu
        imgs = [img0, img1, img2]
        grids = [grid0, grid1, grid2]
        
        # Quan trọng: test_mode=True để lấy kết quả depth cuối cùng
        # iters=12 là số vòng lặp GRU mặc định lúc inference
        return self.model(imgs, grids, iters=5, test_mode=True)

# ==========================================
# 3. MAIN EXPORT FUNCTION
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # Đường dẫn đến file .pth đã train (trên Windows)
    parser.add_argument('--ckpt_path', type=str, default=r"checkpoints/romnistereo32_v19_bs16_e0.pth", help="Path to .pth checkpoint")
    parser.add_argument('--output_path', type=str, default=r"checkpoints/onnx/romnistereo32_v19_bs16_e0_jetson.onnx", help="Output ONNX file")
    args = parser.parse_args()

    # 1. Load Config & Model
    opts = get_opts()
    print(f"Loading model with Base Channel: {opts.net_opts.base_channel}...")
    
    model = ROmniStereo(opts.net_opts)
    
    # 2. Load Weights
    print(f"Loading checkpoint: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    
    # Xử lý key 'module.' nếu train bằng DataParallel
    state_dict = checkpoint['net_state_dict'] if 'net_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Load state dict (strict=False để tránh lỗi buffer nhỏ không khớp, nhưng nên cẩn thận)
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Wrap model để input/output gọn gàng cho ONNX
    onnx_wrapper = ROmniStereoONNX(model)

    # 3. Prepare Dummy Inputs (Theo config cũ của bạn)
    print("Preparing dummy inputs...")
    H_in, W_in = 384, 400
    C_in = 1  # Grayscale
    
    # Images
    img0 = torch.randn(1, C_in, H_in, W_in)
    img1 = torch.randn(1, C_in, H_in, W_in)
    img2 = torch.randn(1, C_in, H_in, W_in)

    # Grids
    # Tính toán shape grid dựa trên config
    # equirect_size=[128, 400], num_downsample=1 => H_out=64, W_out=200
    # num_invdepth=128, num_downsample=1 => D_out=64
    H_out = opts.data_opts.equirect_size[0] // (2 ** opts.data_opts.num_downsample)
    W_out = opts.data_opts.equirect_size[1] // (2 ** opts.data_opts.num_downsample)
    D_out = opts.data_opts.num_invdepth // (2 ** opts.data_opts.num_downsample)
    
    grid_shape = (H_out, W_out, D_out, 2) # Ví dụ: (64, 200, 64, 2)
    print(f"Using dummy grid shape: {grid_shape}")
    
    grid0 = torch.randn(*grid_shape)
    grid1 = torch.randn(*grid_shape)
    grid2 = torch.randn(*grid_shape)

    # 4. Export to ONNX
    print(f"Exporting to {args.output_path} (Opset 13 for TensorRT)...")
    torch.onnx.export(
        onnx_wrapper,
        (img0, img1, img2, grid0, grid1, grid2),
        args.output_path,
        input_names=["img0", "img1", "img2", "grid0", "grid1", "grid2"],
        output_names=["depth_map"],
        
        # --- SỬA DÒNG NÀY ---
        opset_version=11,  # --- Đổi từ 13 thành 11
        # --------------------
        
        do_constant_folding=True,
        keep_initializers_as_inputs=False
    )
    
    print("✅ Export successful! Copy this file to your Jetson Orin Nano.")

if __name__ == "__main__":
    main()