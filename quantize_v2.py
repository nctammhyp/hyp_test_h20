import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
import os
import cv2
import numpy as np
import glob
import json # Thêm thư viện json
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
NUM_CALIB_SAMPLES = 5 
INPUT_SIZE = (800, 768)

# =============================================================================
# 2. WRAPPER & DATA LOADER (GIỮ NGUYÊN)
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
        search_patterns = [os.path.join(img_root, "cam1", "*.jpg"), os.path.join(img_root, "cam1", "*.png"), os.path.join(img_root, "*.jpg"), os.path.join(img_root, "*.png")]
        self.img1_paths = []
        for pattern in search_patterns:
            found = sorted(glob.glob(pattern))
            if len(found) > 0: self.img1_paths = found[:limit]; break
        if len(self.img1_paths) == 0:
            self.use_dummy = True; self.img1_paths = list(range(limit))
        else: self.use_dummy = False
    def preprocess(self, img_path):
        if self.use_dummy: return torch.randn(1, INPUT_SIZE[1], INPUT_SIZE[0])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return torch.zeros(1, INPUT_SIZE[1], INPUT_SIZE[0]).float()
        img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        mean = np.mean(img); std = np.std(img) + 1e-6
        img = (img - mean) / std
        return torch.from_numpy(img).unsqueeze(0).float()
    def __len__(self): return len(self.img1_paths)
    def __getitem__(self, idx):
        if self.use_dummy: p1, p2, p3 = "", "", ""
        else:
            p1 = self.img1_paths[idx]; p2 = p1.replace("cam1", "cam2"); p3 = p1.replace("cam1", "cam3")
            if not os.path.exists(p2): p2 = p1
            if not os.path.exists(p3): p3 = p1
        return self.preprocess(p1), self.preprocess(p2), self.preprocess(p3), self.grids[0], self.grids[1], self.grids[2]

# =============================================================================
# 3. FORWARD FUNCTIONS
# =============================================================================
def adaround_forward_fn(model, batch_data):
    img0, img1, img2, g0, g1, g2 = batch_data
    device = next(model.parameters()).device
    img0, img1, img2 = img0.to(device), img1.to(device), img2.to(device)
    g0, g1, g2 = g0.to(device), g1.to(device), g2.to(device)
    g0 = g0.squeeze(0); g1 = g1.squeeze(0); g2 = g2.squeeze(0)
    return model(img0, img1, img2, g0, g1, g2)

def calibration_callback(model, calib_loader):
    device = next(model.parameters()).device 
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(calib_loader), total=len(calib_loader), leave=False):
            adaround_forward_fn(model, batch)

# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"--- AIMET AdaRound Pipeline on {device} ---")

    # 1. Load Model
    opts = Edict()
    opts.data_opts = Edict({'phi_deg': 45.0, 'num_invdepth': 192, 'equirect_size': [160, 640], 'num_downsample': 1, 'use_rgb': False})
    opts.net_opts = Edict({'base_channel': 32, 'num_invdepth': 192, 'use_rgb': False, 'encoder_downsample_twice': False, 'num_downsample': 1, 'corr_levels': 4, 'corr_radius': 4, 'mixed_precision': False, 'fix_bn': True})

    model = ROmniStereo(opts.net_opts)
    checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    sd = checkpoint['net_state_dict'] if 'net_state_dict' in checkpoint else checkpoint
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device); model.eval()

    # 2. Load Data
    ds_tool = ProjectDataset(DB_NAME, db_opts=Edict({'use_rgb': False, 'num_downsample': 1}), load_lut=False, train=False, db_root=DB_ROOT)
    grids_np = ds_tool.buildLookupTable(output_gpu_tensor=False)
    grids_tensor = [torch.from_numpy(g.astype(np.float32)).to(device) for g in grids_np]
    calib_loader = DataLoader(CalibrationDataset(IMG_ROOT_DIR, grids_tensor, limit=NUM_CALIB_SAMPLES), batch_size=1, shuffle=False)

    # 3. Wrap & Fold BN
    wrapper = ROmniStereoWrapper(model).to(device)
    dummy_input = (torch.randn(1, 1, 768, 800).to(device), torch.randn(1, 1, 768, 800).to(device), torch.randn(1, 1, 768, 800).to(device), grids_tensor[0], grids_tensor[1], grids_tensor[2])
    _ = fold_all_batch_norms(wrapper, input_shapes=[(1,1,768,800)]*3 + [(80,320,96,2)]*3)

    # 4. TRICK: CREATE CONFIG TO IGNORE UPDATE_BLOCK
    # Cách này sẽ bảo AdaRound bỏ qua các layer lặp
    config = {
        "defaults": { "ops": { "is_output_quantized": "True", "is_weight_quantized": "True" },
                      "params": { "is_quantized": "True" } },
        "ops": { "Expand": { "is_output_quantized": "False" } }, # Bỏ qua các node expand
        "op_type": { "Recurrent": { "is_quantized": "False" } }
    }
    
    # Tìm tên chính xác của các layer trong update_block để disable
    layer_specific_rules = {}
    for name, module in wrapper.named_modules():
        if "update_block" in name and isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_specific_rules[name] = { "is_weight_quantized": "False" }
    
    config["layer_name"] = layer_specific_rules
    
    config_file_path = "adaround_config.json"
    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=4)

    # 5. APPLY ADAROUND
    print("4. Applying AdaRound...")
    params = AdaroundParameters(data_loader=calib_loader, num_batches=len(calib_loader), default_num_iterations=10, forward_fn=adaround_forward_fn)
    
    # Ở phiên bản này, ta truyền config_file thay vì ignore_modules
    adarounded_model = Adaround.apply_adaround(
        wrapper, 
        dummy_input, 
        params,
        path=OUTPUT_DIR,
        filename_prefix='adaround',
        default_param_bw=8,
        default_quant_scheme=QuantScheme.post_training_tf,
        config_file=config_file_path # <--- DÙNG FILE CONFIG TẠI ĐÂY
    )
    
    print("✅ AdaRound Complete.")

    # 6. Create QuantSim & Export
    sim = QuantizationSimModel(model=adarounded_model, dummy_input=dummy_input, quant_scheme=QuantScheme.post_training_tf, 
                               default_output_bw=8, default_param_bw=8, config_file=config_file_path)
    sim.compute_encodings(forward_pass_callback=calibration_callback, forward_pass_callback_args=calib_loader)

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    sim.model.to('cpu')
    dummy_cpu = tuple([d.cpu() for d in dummy_input])
    try:
        sim.onnx.export(OUTPUT_DIR, "romni_adaround", dummy_cpu, opset_version=11)
    except:
        sim.export(OUTPUT_DIR, "romni_adaround", dummy_cpu)

    print(f"✅ SUCCESS! Output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()