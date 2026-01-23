import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from easydict import EasyDict as Edict
import sys

# --- AIMET IMPORTS ---
try:
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.batch_norm_fold import fold_all_batch_norms
    from aimet_common.defs import QuantScheme
    # --- ADAROUND IMPORTS ---
    from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
except ImportError:
    print("❌ Error: AIMET is not installed.")
    sys.exit(1)

# --- PROJECT IMPORTS ---
from dataset import Dataset as ProjectDataset
from module.network import ROmniStereo

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
CKPT_PATH = "/home/sw-tamnguyen/Desktop/depth_project/hyp_test_h20/checkpoints/romnistereo32_v6_bs32/romnistereo32_v6_bs32_e40.pth"
DB_ROOT = "/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/hyp_data_01_trainable/"
DB_NAME = "omnithings"
IMG_ROOT_DIR = "/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/hyp_data_01_trainable/omnithings"

OUTPUT_DIR = "./aimet_export_adaround"
# AdaRound cần nhiều dữ liệu hơn để học cách làm tròn tốt nhất (Khuyến nghị: 32 - 128 mẫu)
NUM_CALIB_SAMPLES = 32  
INPUT_SIZE = (800, 768)

# Thiết lập thiết bị (AdaRound chạy rất chậm trên CPU, nên ưu tiên CUDA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 2. WRAPPER & DATA LOADER
# =============================================================================
class ROmniStereoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, img0, img1, img2, grid0, grid1, grid2):
        imgs = [img0, img1, img2]
        grids = [grid0, grid1, grid2]
        # test_mode=True để trả về tensor cuối cùng thay vì list
        return self.model(imgs, grids, iters=12, test_mode=True)

class CalibrationDataset(TorchDataset):
    def __init__(self, img_root, grids, limit=50):
        self.grids = grids
        self.limit = limit
        
        search_patterns = [
            os.path.join(img_root, "cam1", "*.jpg"),
            os.path.join(img_root, "cam1", "*.png"),
            os.path.join(img_root, "*.jpg"),
            os.path.join(img_root, "*.png")
        ]
        self.img1_paths = []
        for pattern in search_patterns:
            found = sorted(glob.glob(pattern))
            if len(found) > 0:
                self.img1_paths = found[:limit]
                break
        
        if len(self.img1_paths) == 0:
            print(f"⚠️  WARNING: No images found. Using dummy data.")
            self.use_dummy = True
            self.img1_paths = list(range(limit))
        else:
            self.use_dummy = False
            print(f"-> Found {len(self.img1_paths)} calibration samples.")

    def preprocess(self, img_path):
        if self.use_dummy:
            return torch.randn(1, INPUT_SIZE[1], INPUT_SIZE[0])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return torch.zeros(1, INPUT_SIZE[1], INPUT_SIZE[0]).float()
        img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        mean = np.mean(img)
        std = np.std(img) + 1e-6
        img = (img - mean) / std
        return torch.from_numpy(img).unsqueeze(0).float()

    def __len__(self): return len(self.img1_paths)

    def __getitem__(self, idx):
        if self.use_dummy:
            path1, path2, path3 = "", "", ""
        else:
            path1 = self.img1_paths[idx]
            path2 = path1.replace("cam1", "cam2")
            path3 = path1.replace("cam1", "cam3")
            if not os.path.exists(path2): path2 = path1
            if not os.path.exists(path3): path3 = path1

        img0 = self.preprocess(path1)
        img1 = self.preprocess(path2)
        img2 = self.preprocess(path3)
        return img0, img1, img2, self.grids[0], self.grids[1], self.grids[2]

# =============================================================================
# 3. CALIBRATION CALLBACK (Cho QuantSim Activations)
# =============================================================================
def calibration_callback(model, calib_loader):
    model.eval()
    # model.to(DEVICE) # Model đã ở device do AIMET quản lý
    
    print(f"   -> Running Forward Pass for Activation Calibration...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(calib_loader), total=len(calib_loader)):
            # Chuyển dữ liệu sang device
            batch = [t.to(DEVICE) for t in batch]
            img0, img1, img2, g0, g1, g2 = batch
            
            # Grids cần squeeze nếu loader tạo thêm dim
            if g0.dim() > 4: # batch_size, H, W, D, 2
                g0 = g0.squeeze(0)
                g1 = g1.squeeze(0)
                g2 = g2.squeeze(0)
            
            model(img0, img1, img2, g0, g1, g2)

# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    print(f"--- AIMET AdaRound Quantization Pipeline on {DEVICE} ---")

    # 1. Load Model
    print("1. Loading Model...")
    opts = Edict()
    opts.data_opts = Edict({'phi_deg': 45.0, 'num_invdepth': 192, 'equirect_size': [160, 640], 'num_downsample': 1, 'use_rgb': False})
    opts.net_opts = Edict({'base_channel': 32, 'num_invdepth': 192, 'use_rgb': False, 'encoder_downsample_twice': False, 'num_downsample': 1, 'corr_levels': 4, 'corr_radius': 4, 'mixed_precision': False, 'fix_bn': True})

    model = ROmniStereo(opts.net_opts)
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
        sd = checkpoint['net_state_dict'] if 'net_state_dict' in checkpoint else checkpoint
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        print(f"❌ Checkpoint not found: {CKPT_PATH}"); return

    # Chuyển model sang device (ưu tiên GPU cho AdaRound)
    model.to(DEVICE); model.eval()

    # 2. Load Grids
    print("2. Loading Grids...")
    dummy_opts = Edict({'use_rgb': False, 'num_downsample': 1})
    ds_tool = ProjectDataset(DB_NAME, db_opts=dummy_opts, load_lut=False, train=False, db_root=DB_ROOT)
    grids_np = ds_tool.buildLookupTable(output_gpu_tensor=False)
    # Load grids lên device
    grids_tensor = [torch.from_numpy(g.astype(np.float32)).to(DEVICE) for g in grids_np]

    # 3. Prepare Data Loader
    print("3. Preparing Data Loader...")
    # AdaRound yêu cầu batch_size (thường là 1 cho các model phức tạp để tiết kiệm VRAM)
    calib_ds = CalibrationDataset(IMG_ROOT_DIR, grids_tensor, limit=NUM_CALIB_SAMPLES)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False)

    # 4. Wrap & Fold BN
    print("4. Wrapping & Folding BN...")
    wrapper = ROmniStereoWrapper(model).to(DEVICE)
    wrapper.eval()
    
    # Dummy input trên Device
    dummy_img = torch.randn(1, 1, 768, 800).to(DEVICE)
    dummy_grid = grids_tensor[0]
    dummy_input = (dummy_img, dummy_img, dummy_img, dummy_grid, dummy_grid, dummy_grid)

    _ = fold_all_batch_norms(wrapper, input_shapes=[
        (1, 1, 768, 800), (1, 1, 768, 800), (1, 1, 768, 800),
        (80, 320, 96, 2), (80, 320, 96, 2), (80, 320, 96, 2)
    ])

    # =========================================================================
    # 5. APPLY ADAROUND (Bước mới)
    # =========================================================================
    print("5. Applying AdaRound (Adaptive Rounding)...")
    print("   Note: This process is computationally expensive and takes time.")

    # Định nghĩa tham số AdaRound
    params = AdaroundParameters(
        data_loader=calib_loader,
        num_batches=len(calib_loader),  # Sử dụng toàn bộ dữ liệu trong loader
        default_num_iterations=10000,   # 10k iters là chuẩn (có thể giảm xuống 2000-5000 để test nhanh)
        default_reg_param=0.01,
        default_beta_range=(20, 2),
        default_warm_start=0.2
    )

    # Chạy AdaRound: Trả về model với Weights đã được tối ưu
    # path: thư mục để lưu log hoặc cache nếu cần
    adarounded_model = Adaround.apply_adaround(
        wrapper, 
        dummy_input, 
        params,
        path=OUTPUT_DIR,
        filename_prefix='adaround',
        default_param_bw=8,
        default_quant_scheme=QuantScheme.post_training_tf # TF scheme thường tốt cho DSP/NPU
    )

    print("✅ AdaRound Complete!")

    # =========================================================================
    # 6. CREATE QUANTSIM (Từ model đã AdaRound)
    # =========================================================================
    print("6. Creating QuantSim using AdaRounded Model...")
    
    # Lưu ý: Chúng ta dùng adarounded_model ở đây
    sim = QuantizationSimModel(
        model=adarounded_model,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.post_training_tf,
        default_output_bw=8,
        default_param_bw=8
    )

    # =========================================================================
    # 7. COMPUTE ACTIVATION ENCODINGS
    # =========================================================================
    print("7. Computing Activation Encodings...")
    # Chúng ta dùng lại loader cũ cho việc calibrate activation
    sim.compute_encodings(forward_pass_callback=calibration_callback, forward_pass_callback_args=calib_loader)

    # =========================================================================
    # 8. EXPORT
    # =========================================================================
    print(f"8. Exporting to {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    del calib_loader, calib_ds, ds_tool
    import gc; gc.collect()

    try:
        sim.onnx.export(
            output_dir=OUTPUT_DIR,
            filename_prefix="romni_adarounded",
            dummy_input=dummy_input,
            opset_version=11
        )
        print("\n✅ DONE! Exported using sim.onnx.export (Opset 11)")
        
    except (TypeError, AttributeError):
        print("⚠️ Warning: sim.onnx.export failed, falling back to sim.export...")
        sim.export(
            path=OUTPUT_DIR,
            filename_prefix="romni_adarounded",
            dummy_input=dummy_input
        )
        print("\n✅ DONE! Exported using sim.export")

    print(f"  - ONNX: {os.path.join(OUTPUT_DIR, 'romni_adarounded.onnx')}")
    print(f"  - Encodings: {os.path.join(OUTPUT_DIR, 'romni_adarounded.encodings')}")

if __name__ == "__main__":
    main()