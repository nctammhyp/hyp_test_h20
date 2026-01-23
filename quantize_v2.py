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

# --- AIMET IMPORTS ---
try:
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.batch_norm_fold import fold_all_batch_norms
    from aimet_common.defs import QuantScheme
    from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
except ImportError:
    print("❌ Error: AIMET is not installed.")
    sys.exit(1)

from dataset import Dataset as ProjectDataset
from module.network import ROmniStereo

# =============================================================================
# CONFIGURATION
# =============================================================================
CKPT_PATH = "/home/sw-tamnguyen/Desktop/depth_project/hyp_test_h20/checkpoints/romnistereo32_v6_bs32/romnistereo32_v6_bs32_e40.pth"
DB_ROOT = "/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/hyp_data_01_trainable/"
DB_NAME = "omnithings"
IMG_ROOT_DIR = "/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/hyp_data_01_trainable/omnithings"
OUTPUT_DIR = "./aimet_export_adaround"
NUM_CALIB_SAMPLES = 5 
INPUT_SIZE = (800, 768)

# =============================================================================
# WRAPPER & DATA LOADER
# =============================================================================
class ROmniStereoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, img0, img1, img2, grid0, grid1, grid2):
        return self.model([img0, img1, img2], [grid0, grid1, grid2], iters=12, test_mode=True)

class CalibrationDataset(TorchDataset):
    def __init__(self, img_root, grids, limit=50):
        self.grids = grids
        self.limit = limit
        self.img1_paths = sorted(glob.glob(os.path.join(img_root, "cam1", "*.jpg")))[:limit]
        if not self.img1_paths: self.img1_paths = sorted(glob.glob(os.path.join(img_root, "*.jpg")))[:limit]
    def preprocess(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return torch.zeros(1, INPUT_SIZE[1], INPUT_SIZE[0])
        img = cv2.resize(img, INPUT_SIZE)
        img = (img.astype(np.float32) - np.mean(img)) / (np.std(img) + 1e-6)
        return torch.from_numpy(img).unsqueeze(0)
    def __len__(self): return len(self.img1_paths)
    def __getitem__(self, i):
        p1 = self.img1_paths[i]; p2 = p1.replace("cam1", "cam2"); p3 = p1.replace("cam1", "cam3")
        return self.preprocess(p1), self.preprocess(p2), self.preprocess(p3), self.grids[0], self.grids[1], self.grids[2]

def adaround_forward_fn(model, batch_data):
    img0, img1, img2, g0, g1, g2 = [x.to(next(model.parameters()).device) for x in batch_data]
    return model(img0, img1, img2, g0.squeeze(0), g1.squeeze(0), g2.squeeze(0))

def calibration_callback(model, calib_loader):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(calib_loader, desc="Calibrating"): adaround_forward_fn(model, batch)

# =============================================================================
# MAIN
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- AIMET AdaRound Pipeline on {device} ---")

    # 1. Load Model
    opts = Edict({'base_channel': 32, 'num_invdepth': 192, 'use_rgb': False, 'encoder_downsample_twice': False, 'num_downsample': 1, 'corr_levels': 4, 'corr_radius': 4, 'mixed_precision': False, 'fix_bn': True})
    model = ROmniStereo(opts)
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    sd = checkpoint['net_state_dict'] if 'net_state_dict' in checkpoint else checkpoint
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    model.to(device).eval()

    # 2. Data
    ds_tool = ProjectDataset(DB_NAME, db_opts=Edict({'use_rgb': False, 'num_downsample': 1}), load_lut=False, train=False, db_root=DB_ROOT)
    grids = [torch.from_numpy(g.astype(np.float32)).to(device) for g in ds_tool.buildLookupTable(False)]
    calib_loader = DataLoader(CalibrationDataset(IMG_ROOT_DIR, grids, NUM_CALIB_SAMPLES), batch_size=1)

    # 3. Wrap & BN Folding
    wrapper = ROmniStereoWrapper(model)
    dummy_input = (torch.randn(1, 1, 768, 800).to(device), torch.randn(1, 1, 768, 800).to(device), torch.randn(1, 1, 768, 800).to(device), grids[0], grids[1], grids[2])
    _ = fold_all_batch_norms(wrapper, input_shapes=[(1,1,768,800)]*3 + [(80,320,96,2)]*3)

    # 4. CREATE CONFIG JSON (Để loại bỏ UpdateBlock khỏi AdaRound)
    config = {
        "defaults": {"ops": {"is_output_quantized": "True", "is_weight_quantized": "True"}, "params": {"is_quantized": "True"}},
        "layer_name": {}
    }
    # Tắt AdaRound cho các layer lặp trong update_block
    for name, module in wrapper.named_modules():
        if "update_block" in name and isinstance(module, (nn.Conv2d, nn.Linear)):
            config["layer_name"][name] = {"is_weight_quantized": "False"}
    
    config_path = "adaround_config.json"
    with open(config_path, "w") as f: json.dump(config, f, indent=4)

    # 5. APPLY ADAROUND
    print("4. Applying AdaRound...")
    params = AdaroundParameters(data_loader=calib_loader, num_batches=len(calib_loader), default_num_iterations=10, forward_fn=adaround_forward_fn)
    
    # SỬA TÊN THAM SỐ: default_config_file thay vì config_file
    adarounded_model = Adaround.apply_adaround(
        wrapper, dummy_input, params, path=OUTPUT_DIR, filename_prefix='adaround',
        default_param_bw=8, default_quant_scheme=QuantScheme.post_training_tf,
        default_config_file=config_path # <--- ĐÃ SỬA TẠI ĐÂY
    )

    # 6. QuantSim & Export
    sim = QuantizationSimModel(adarounded_model, dummy_input, QuantScheme.post_training_tf, config_file=config_path)
    sim.compute_encodings(calibration_callback, calib_loader)

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    sim.model.to('cpu')
    try:
        sim.onnx.export(OUTPUT_DIR, "romni_adaround", tuple([d.cpu() for d in dummy_input]), opset_version=11)
    except:
        sim.export(OUTPUT_DIR, "romni_adaround", tuple([d.cpu() for d in dummy_input]))

    print(f"✅ SUCCESS! Output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()