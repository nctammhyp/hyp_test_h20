import torch
import torch.nn as nn
import argparse
import os
from easydict import EasyDict as Edict
import numpy as np

# Import model của bạn
from module.network import ROmniStereo

# ==========================================
# 1. Wrapper Class để làm phẳng Input
# ==========================================
class ROmniStereoONNX(nn.Module):
    """
    ONNX không hỗ trợ input là list/tuple tốt.
    Wrapper này nhận các tensor rời rạc và gom lại thành list để model gốc xử lý.
    """
    def __init__(self, model, iters=12):
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, img0, img1, img2, grid0, grid1, grid2):
        # 1. Gom input thành list như model gốc yêu cầu
        imgs = [img0, img1, img2]
        grids = [grid0, grid1, grid2]
        
        # 2. Gọi forward với test_mode=True để lấy kết quả cuối cùng
        # Output sẽ là disparity map (tensor)
        final_inv_depth = self.model(imgs, grids, iters=self.iters, test_mode=True)
        
        return final_inv_depth

# ==========================================
# 2. Cấu hình & Load Model
# ==========================================
def get_opts(args):
    # Cấu hình giống hệt lúc train
    opts = Edict()
    opts.data_opts = Edict({
        'phi_deg': 45.0, 
        'num_invdepth': 192, 
        'equirect_size': [160, 640], # [H, W] output
        'num_downsample': 1, 
        'use_rgb': False
    })
    opts.net_opts = Edict({
        'base_channel': 32, 
        'num_invdepth': 192, 
        'use_rgb': False, 
        'encoder_downsample_twice': False, 
        'num_downsample': 1, 
        'corr_levels': 4, 
        'corr_radius': 4, 
        'mixed_precision': False, 
        'fix_bn': False # Thường là False khi eval/export
    })
    return opts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to .pth/.pt file")
    parser.add_argument('--output_path', type=str, default="romnistereo.onnx")
    parser.add_argument('--iters', type=int, default=12, help="Number of GRU iterations")
    args = parser.parse_args()

    device = torch.device("cpu") # Export trên CPU cho ổn định (tránh lỗi memory)
    
    # 1. Khởi tạo Model gốc
    opts = get_opts(args)
    model = ROmniStereo(opts.net_opts)
    
    # 2. Load Weights
    print(f"Loading checkpoint from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    
    # Xử lý key 'module.' nếu train bằng DataParallel
    if 'net_state_dict' in checkpoint:
        state_dict = checkpoint['net_state_dict']
    else:
        state_dict = checkpoint # Trường hợp save trực tiếp state_dict

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 3. Bọc model bằng Wrapper ONNX
    onnx_model = ROmniStereoONNX(model, iters=args.iters)
    onnx_model.eval()

    # ==========================================
    # 3. Tạo Dummy Inputs (Dữ liệu giả lập)
    # ==========================================
    print("Creating dummy inputs...")
    
    # Kích thước ảnh đầu vào (dựa trên dataset code: resize về 800x768)
    # Shape: (Batch, Channel, Height, Width)
    H_in, W_in = 768, 800
    C_in = 3 if opts.net_opts.use_rgb else 1
    batch_size = 1

    # 3 Ảnh đầu vào (Cam 1, 2, 3)
    img0 = torch.randn(batch_size, C_in, H_in, W_in).to(device)
    img1 = torch.randn(batch_size, C_in, H_in, W_in).to(device)
    img2 = torch.randn(batch_size, C_in, H_in, W_in).to(device)

    # 3 Grids (Lookup Tables)
    # Tính kích thước Grid dựa trên equirect_size và downsample
    # Grid shape logic từ dataset.py: [H_out, W_out, Num_Depth, 2]
    H_out = opts.data_opts.equirect_size[0] // (2 ** opts.data_opts.num_downsample)
    W_out = opts.data_opts.equirect_size[1] // (2 ** opts.data_opts.num_downsample)
    D_out = opts.data_opts.num_invdepth // (2 ** opts.data_opts.num_downsample)
    
    # Grid trong Dataset code là numpy, vào model là Tensor
    # Lưu ý: Model grid_sample yêu cầu grid float
    grid_shape = (H_out, W_out, D_out, 2) 
    print(f"Grid shape expected: {grid_shape}")
    
    grid0 = torch.randn(*grid_shape).to(device)
    grid1 = torch.randn(*grid_shape).to(device)
    grid2 = torch.randn(*grid_shape).to(device)

    # ==========================================
    # 4. Export ONNX
    # ==========================================
    print(f"Exporting to {args.output_path}...")
    
    input_names = ["img0", "img1", "img2", "grid0", "grid1", "grid2"]
    output_names = ["inverse_depth_map"]
    
    # Dynamic axes: Để cho phép batch size thay đổi lúc inference
    dynamic_axes = {
        "img0": {0: "batch_size"},
        "img1": {0: "batch_size"},
        "img2": {0: "batch_size"},
        "inverse_depth_map": {0: "batch_size"}
    }

    torch.onnx.export(
        onnx_model,
        (img0, img1, img2, grid0, grid1, grid2), # Tuple inputs
        args.output_path,
        export_params=True,
        opset_version=12, # Khuyến nghị >= 11 để hỗ trợ grid_sample tốt nhất
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print("✅ Export success!")
    print(f"Inputs: {input_names}")
    print(f"Output: {output_names}")
    
    # (Optional) Verify ONNX
    try:
        import onnx
        onnx_model_proto = onnx.load(args.output_path)
        onnx.checker.check_model(onnx_model_proto)
        print("✅ ONNX Model check passed.")
    except ImportError:
        print("Skipping ONNX check (onnx library not installed).")

if __name__ == "__main__":
    main()