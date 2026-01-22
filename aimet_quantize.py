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
except ImportError:
    print("❌ Error: AIMET is not installed.")
    sys.exit(1)

# --- PROJECT IMPORTS ---
from dataset import Dataset as ProjectDataset
from module.network import ROmniStereo

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# CKPT_PATH = r"F:\algo\mvs_v119\checkpoints\romnistereo32_v6_bs32_e37.pth"
# DB_ROOT = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable"
# DB_NAME = "omnithings"
# IMG_ROOT_DIR = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable\omnithings"

CKPT_PATH = "/home/sw-tamnguyen/Desktop/depth_project/hyp_test_h20/checkpoints/romnistereo32_v6_bs32/romnistereo32_v6_bs32_e40.pth"
DB_ROOT = "/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/hyp_data_01_trainable/"
DB_NAME = "omnithings"
IMG_ROOT_DIR = "/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/hyp_data_01_trainable/omnithings"

OUTPUT_DIR = "./aimet_export"
NUM_CALIB_SAMPLES = 5 
INPUT_SIZE = (800, 768)

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
# 3. CALIBRATION CALLBACK
# =============================================================================
def calibration_callback(model, calib_loader):
    # Ép chạy CPU trong callback
    device = torch.device("cpu")
    model.eval()
    # model.to(device) # Model thường đã ở device đúng do AIMET quản lý, nhưng cứ chắc chắn
    
    print(f"   -> Running Forward Pass for Calibration...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(calib_loader), total=len(calib_loader)):
            img0, img1, img2, g0, g1, g2 = batch
            
            # Đảm bảo input ở CPU
            img0, img1, img2 = img0.to(device), img1.to(device), img2.to(device)
            g0, g1, g2 = g0.to(device), g1.to(device), g2.to(device)

            g0 = g0.squeeze(0)
            g1 = g1.squeeze(0)
            g2 = g2.squeeze(0)
            
            model(img0, img1, img2, g0, g1, g2)

# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    device = torch.device("cpu") # ALL CPU
    print(f"--- AIMET Quantization Pipeline on {device} ---")

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

    model.to(device); model.eval()

    # 2. Load Grids
    print("2. Loading Grids...")
    dummy_opts = Edict({'use_rgb': False, 'num_downsample': 1})
    ds_tool = ProjectDataset(DB_NAME, db_opts=dummy_opts, load_lut=False, train=False, db_root=DB_ROOT)
    grids_np = ds_tool.buildLookupTable(output_gpu_tensor=False)
    grids_tensor = [torch.from_numpy(g.astype(np.float32)).to(device) for g in grids_np]

    # 3. Prepare Data Loader
    print("3. Preparing Data Loader...")
    calib_ds = CalibrationDataset(IMG_ROOT_DIR, grids_tensor, limit=NUM_CALIB_SAMPLES)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False)

    # 4. Wrap & Fold BN
    print("4. Wrapping & Folding BN...")
    wrapper = ROmniStereoWrapper(model).to(device)
    
    # Dummy input trên CPU
    dummy_img = torch.randn(1, 1, 768, 800).to(device)
    dummy_grid = grids_tensor[0]
    dummy_input = (dummy_img, dummy_img, dummy_img, dummy_grid, dummy_grid, dummy_grid)

    _ = fold_all_batch_norms(wrapper, input_shapes=[
        (1, 1, 768, 800), (1, 1, 768, 800), (1, 1, 768, 800),
        (80, 320, 96, 2), (80, 320, 96, 2), (80, 320, 96, 2)
    ])

    # 5. Create QuantSim
    print("5. Creating QuantSim...")
    sim = QuantizationSimModel(
        model=wrapper,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.post_training_tf,
        default_output_bw=8,
        default_param_bw=8
    )

    # 6. Compute Encodings
    print("6. Computing Encodings...")
    sim.compute_encodings(forward_pass_callback=calibration_callback, forward_pass_callback_args=calib_loader)

    # 7. Export (QUAY VỀ HÀM CHUẨN - IGNORE WARNING)
    print(f"7. Exporting to {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # Sử dụng sim.export truyền thống
    # Hàm này sẽ gọi torch.onnx.export bên trong.
    # Vì chúng ta đang chạy CPU, nó sẽ KHÔNG bị lỗi Device Mismatch.
    sim.export(
        path=OUTPUT_DIR,
        filename_prefix="romni_quantized",
        dummy_input=dummy_input
    )
    
    print("\n✅ DONE! You should see .onnx and .encodings files in aimet_export.")

if __name__ == "__main__":
    main()