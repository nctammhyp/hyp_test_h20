import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
import os
import cv2
import numpy as np
import glob
import json
from tqdm import tqdm
from easydict import EasyDict as Edict
import sys
import gc

# --- AIMET IMPORTS ---
try:
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.batch_norm_fold import fold_all_batch_norms
    from aimet_common.defs import QuantScheme
    from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
except ImportError:
    print("❌ Error: AIMET is not installed. Please install 'aimet-torch'.")
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
NUM_CALIB_SAMPLES = 50 # Tăng lên 50-100 để AdaRound học chính xác hơn
INPUT_SIZE = (800, 768)

# =============================================================================
# 2. WRAPPER & DATA LOADER
# =============================================================================
class ROmniStereoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, img0, img1, img2, grid0, grid1, grid2):
        # Chuyển list sang cấu trúc model gốc mong đợi
        return self.model([img0, img1, img2], [grid0, grid1, grid2], iters=12, test_mode=True)

class CalibrationDataset(TorchDataset):
    def __init__(self, img_root, grids, limit=50):
        self.grids = grids
        self.limit = limit
        # Tìm ảnh mẫu cam1
        search_path = os.path.join(img_root, "cam1", "*.jpg")
        self.img1_paths = sorted(glob.glob(search_path))[:limit]
        if not self.img1_paths:
            self.img1_paths = sorted(glob.glob(os.path.join(img_root, "*.jpg")))[:limit]
        print(f"-> Found {len(self.img1_paths)} samples for AdaRound.")

    def preprocess(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return torch.zeros(1, INPUT_SIZE[1], INPUT_SIZE[0]).float()
        img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        # Normalize logic chuẩn
        img = (img.astype(np.float32) - np.mean(img)) / (np.std(img) + 1e-6)
        return torch.from_numpy(img).unsqueeze(0)

    def __len__(self): return len(self.img1_paths)

    def __getitem__(self, idx):
        p1 = self.img1_paths[idx]
        p2 = p1.replace("cam1", "cam2")
        p3 = p1.replace("cam1", "cam3")
        if not os.path.exists(p2): p2 = p1
        if not os.path.exists(p3): p3 = p1
        return self.preprocess(p1), self.preprocess(p2), self.preprocess(p3), self.grids[0], self.grids[1], self.grids[2]

# =============================================================================
# 3. FORWARD HELPERS
# =============================================================================
def adaround_forward_fn(model, batch_data):
    """ Hàm này giải quyết lỗi 'too many values to unpack' """
    img0, img1, img2, g0, g1, g2 = batch_data
    # Lấy device từ model
    dev = next(model.parameters()).device
    
    img0, img1, img2 = img0.to(dev), img1.to(dev), img2.to(dev)
    g0, g1, g2 = g0.to(dev), g1.to(dev), g2.to(dev)
    
    # Squeeze batch dimension cho Grid (vì model nhận Rank 4)
    return model(img0, img1, img2, g0.squeeze(0), g1.squeeze(0), g2.squeeze(0))

def calibration_callback(model, calib_loader):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(calib_loader, desc="Encoding Calibration", leave=False):
            adaround_forward_fn(model, batch)

# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- AIMET AdaRound Pipeline on {device} ---")

    # 1. Load Model
    print("1. Loading Model...")
    opts = Edict({'base_channel': 32, 'num_invdepth': 192, 'use_rgb': False, 'encoder_downsample_twice': False, 'num_downsample': 1, 'corr_levels': 4, 'corr_radius': 4, 'mixed_precision': False, 'fix_bn': True})
    model = ROmniStereo(opts)
    
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    sd = checkpoint['net_state_dict'] if 'net_state_dict' in checkpoint else checkpoint
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    model.to(device).eval()

    # 2. Load Data & Grids (Fix lỗi AttributeErrorgeometry)
    print("2. Computing Calibration Grids...")
    ds_tool = ProjectDataset(DB_NAME, db_opts=Edict({'use_rgb': False, 'num_downsample': 1}), load_lut=False, train=False, db_root=DB_ROOT)
    # SỬA TẠI ĐÂY: Dùng keyword argument output_gpu_tensor=False
    grids_np = ds_tool.buildLookupTable(output_gpu_tensor=False)
    grids_tensor = [torch.from_numpy(g.astype(np.float32)) for g in grids_np]

    calib_ds = CalibrationDataset(IMG_ROOT_DIR, grids_tensor, limit=NUM_CALIB_SAMPLES)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False)

    # 3. Wrap & Fold BN
    wrapper = ROmniStereoWrapper(model).to(device)
    dummy_input = (torch.randn(1, 1, 768, 800).to(device), 
                   torch.randn(1, 1, 768, 800).to(device), 
                   torch.randn(1, 1, 768, 800).to(device), 
                   grids_tensor[0].to(device), grids_tensor[1].to(device), grids_tensor[2].to(device))

    print("3. Folding Batch Norms...")
    _ = fold_all_batch_norms(wrapper, input_shapes=[(1, 1, 768, 800)]*3 + [(80, 320, 96, 2)]*3)

    # 4. CREATE CONFIG JSON (Fix lỗi KeyError vòng lặp)
    print("4. Creating AIMET Config to ignore recurrent layers...")
    config = {
        "defaults": {
            "ops": {"is_output_quantized": "True", "is_weight_quantized": "True"},
            "params": {"is_quantized": "True"}
        },
        "layer_name": {}
    }
    # Tắt AdaRound cho các layer trong update_block để tránh lỗi mapping do vòng lặp
    for name, module in wrapper.named_modules():
        if "update_block" in name and isinstance(module, (nn.Conv2d, nn.Linear)):
            config["layer_name"][name] = {"is_weight_quantized": "False"}
    
    config_path = "adaround_config_final.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # 5. APPLY ADAROUND
    print("5. Applying AdaRound weight optimization...")
    params = AdaroundParameters(
        data_loader=calib_loader, 
        num_batches=len(calib_loader), 
        default_num_iterations=1000, 
        forward_fn=adaround_forward_fn
    )
    
    # Sử dụng default_config_file để pass config
    adarounded_model = Adaround.apply_adaround(
        wrapper, 
        dummy_input, 
        params, 
        path=OUTPUT_DIR, 
        filename_prefix='adaround_weights',
        default_param_bw=8, 
        default_quant_scheme=QuantScheme.post_training_tf,
        default_config_file=config_path # <--- KEYWORD CHUẨN CHO ADAROUND
    )
    
    print("✅ AdaRound Complete.")

    # 6. QUANTSIM & ENCODINGS
    print("6. Creating QuantSim & Calibrating Activations...")
    # Dùng config_file cho QuantSim
    sim = QuantizationSimModel(
        model=adarounded_model, 
        dummy_input=dummy_input, 
        quant_scheme=QuantScheme.post_training_tf, 
        default_output_bw=8, 
        default_param_bw=8, 
        config_file=config_path # <--- KEYWORD CHUẨN CHO QUANTSIM
    )
    
    sim.compute_encodings(forward_pass_callback=calibration_callback, forward_pass_callback_args=calib_loader)

    # 7. EXPORT
    print(f"7. Exporting to {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # Giải phóng GPU trước khi export để tránh MemoryError
    sim.model.to('cpu')
    dummy_input_cpu = tuple([d.cpu() for d in dummy_input])
    gc.collect(); torch.cuda.empty_cache()

    try:
        # Cố gắng xuất ra Opset 11 để tránh lỗi Unsqueeze trên Qualcomm HTP
        sim.onnx.export(
            output_dir=OUTPUT_DIR,
            filename_prefix="romni_adaround_final",
            dummy_input=dummy_input_cpu,
            opset_version=11
        )
        print("\n✅ DONE! Successfully exported Opset 11 ONNX + Encodings.")
    except Exception as e:
        print(f"⚠️ sim.onnx.export failed, using fallback sim.export: {e}")
        sim.export(path=OUTPUT_DIR, filename_prefix="romni_adaround_final", dummy_input=dummy_input_cpu)

    print(f"Check results in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()