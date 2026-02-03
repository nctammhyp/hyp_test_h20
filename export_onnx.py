import torch
import torch.nn as nn
import argparse
import os
from easydict import EasyDict as Edict
import numpy as np
import onnx

# Import model của bạn
from module.network import ROmniStereo

class ROmniStereoONNX(nn.Module):
    def __init__(self, model, iters=5):
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, img0, img1, img2, grid0, grid1, grid2):
        # 1. Gom input thành list
        imgs = [img0, img1, img2]
        grids = [grid0, grid1, grid2]
        # 2. Gọi forward với test_mode=True
        final_inv_depth = self.model(imgs, grids, iters=self.iters, test_mode=True)
        return final_inv_depth

def get_opts():
    opts = Edict()
    opts.data_opts = Edict({'phi_deg': 45.0, 'num_invdepth': 128, 'equirect_size': [128, 400], 'num_downsample': 1, 'use_rgb': False})
    opts.net_opts = Edict({'base_channel': 16, 'num_invdepth': 128, 'use_rgb': False, 'encoder_downsample_twice': False, 'num_downsample': 1, 'corr_levels': 4, 'corr_radius': 4, 'mixed_precision': False, 'fix_bn': False})
    return opts

def main():
    # Tối ưu hóa JIT cho việc export
    import torch._C
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=r"checkpoints/romnistereo32_v13_bs16_e194.pth")
    parser.add_argument('--output_path', type=str, default=r"checkpoints/onnx/romnistereo32_v13_bs16_e194.onnx")
    parser.add_argument('--iters', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cpu")
    opts = get_opts()
    model = ROmniStereo(opts.net_opts)
    
    print(f"Loading checkpoint: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['net_state_dict'] if 'net_state_dict' in checkpoint else checkpoint
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    model.eval()

    onnx_model_wrapper = ROmniStereoONNX(model, iters=args.iters)
    onnx_model_wrapper.eval()

    # --- DUMMY INPUTS ---
    # H_in, W_in = 768, 800
    H_in, W_in = 384, 400

    C_in = 1 # Grayscale
    
    img0 = torch.randn(1, C_in, H_in, W_in)
    img1 = torch.randn(1, C_in, H_in, W_in)
    img2 = torch.randn(1, C_in, H_in, W_in)

    H_out = opts.data_opts.equirect_size[0] // 2
    W_out = opts.data_opts.equirect_size[1] // 2
    D_out = 64 # num_invdepth // 2
    grid_shape = (H_out, W_out, D_out, 2) # (80, 320, 64, 2)

    print(f"Using dummy grid shape: {grid_shape}")
    
    grid0 = torch.randn(*grid_shape)
    grid1 = torch.randn(*grid_shape)
    grid2 = torch.randn(*grid_shape)

    input_names = ["img0", "img1", "img2", "grid0", "grid1", "grid2"]
    output_names = ["inverse_depth_map"]

    print(f"Exporting to {args.output_path} (Opset 11)...")
    
    # Export thực tế
    torch.onnx.export(
        onnx_model_wrapper,
        (img0, img1, img2, grid0, grid1, grid2),
        args.output_path,
        export_params=True,
        opset_version=18, # Quan trọng cho HTP
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        keep_initializers_as_inputs=False
    )

    # --- HẬU XỬ LÝ: KIỂM TRA VÀ FIX ĐƯỜNG DẪN EXTERNAL DATA ---
    # Nếu file > 2GB, PyTorch sẽ tự tách file .data
    print("Verifying ONNX storage...")
    loaded_model = onnx.load(args.output_path)
    
    # Ép buộc thuộc tính allowzero của Reshape về 0 (Sửa lỗi QAIRT)
    for node in loaded_model.graph.node:
        if node.op_type == "Reshape":
            for attr in node.attribute:
                if attr.name == "allowzero":
                    attr.i = 0
    
    # Lưu lại để đảm bảo tính nhất quán (Nếu > 2GB sẽ tự sinh file .data)
    onnx.save(loaded_model, args.output_path)
    
    print(f"✅ SUCCESS! Final ONNX saved at: {os.path.abspath(args.output_path)}")
    if os.path.exists(args.output_path + ".data"):
        print(f"📦 Large model detected. Also copy this file to Linux: {args.output_path}.data")

if __name__ == "__main__":
    main()