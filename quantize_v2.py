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
NUM_CALIB_SAMPLES = 32  
INPUT_SIZE = (800, 768)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 2. WRAPPER (UPDATED)
# =============================================================================
class ROmniStereoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # --- SỬA ĐỔI QUAN TRỌNG ---
    # Thay vì nhận 6 tham số rời rạc, nhận 1 tuple chứa tất cả inputs
    # Điều này giúp tương thích với logic unpack của AIMET
    def forward(self, inputs_tuple):
        # Unpack tuple
        img0, img1, img2, grid0, grid1, grid2 = inputs_tuple
        
        # Xử lý Grid dimension: Nếu grid có shape (B, H, W, D, 2) -> (B, H, W, D, 2)
        # Model gốc có thể mong đợi grid không có batch dim nếu list grid được lặp qua
        # Nhưng ở đây ta giữ batch dim vì DataLoader tạo ra nó.
        # Lưu ý: Code gốc sph_feats = F.grid_sample(feat[..., None], grid.repeat(bs, 1, 1, 1, 1)...)
        # Nếu grid đã có batch dim từ DataLoader, ta cần kiểm tra logic bên trong model.
        # Tuy nhiên, để an toàn với code lượng tử hóa, ta thường xử lý ở đây.
        
        # Grid từ DataLoader (Batch=1) sẽ là: (1, 80, 320, 96, 2)
        # Model network.py dòng 76: grids_pad = ... torch.cat(..., grid)
        # Model expect grid là tensor (H, W, D, 2) hay (B, H, W, D, 2)?
        # Trong train_u.py: grids = [torch.tensor(grid)...] -> Shape (H, W, D, 2)
        # Khi forward: sph_feats = ... grid.repeat(bs, 1, 1, 1, 1)
        # => Model mong đợi grid KHÔNG CÓ batch dimension (hoặc nó tự repeat).
        
        # FIX: Nếu grid có batch dim = 1, squeeze nó đi để giống behavior lúc training
        if grid0.dim() == 5 and grid0.shape[0] == 1:
            grid0 = grid0.squeeze(0)
            grid1 = grid1.squeeze(0)
            grid2 = grid2.squeeze(0)
            
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
        
        # Gom tất cả inputs vào 1 tuple
        inputs = (img0, img1, img2, self.grids[0], self.grids[1], self.grids[2])
        
        # --- SỬA ĐỔI QUAN TRỌNG ---
        # Trả về (inputs, label). Label là dummy (0).
        # AIMET sẽ unpack: inputs, _ = batch -> Thành công!
        return inputs, 0

# =============================================================================
# 3. CALIBRATION CALLBACK (UPDATED)
# =============================================================================
def calibration_callback(model, calib_loader):
    model.eval()
    # model.to(DEVICE) 
    
    print(f"   -> Running Forward Pass for Activation Calibration...")
    with torch.no_grad():
        for batch in tqdm(calib_loader):
            # Unpack batch từ DataLoader (Inputs, Label)
            inputs, _ = batch 
            
            # Chuyển từng tensor trong tuple inputs sang device
            # inputs lúc này là một list các tensor (do DataLoader collate_fn tạo ra)
            inputs = [t.to(DEVICE) for t in inputs]
            
            # Truyền list inputs vào model (Wrapper sẽ nhận nó như 1 tham số inputs_tuple)
            model(inputs)

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

    model.to(DEVICE); model.eval()

    # 2. Load Grids
    print("2. Loading Grids...")
    dummy_opts = Edict({'use_rgb': False, 'num_downsample': 1})
    ds_tool = ProjectDataset(DB_NAME, db_opts=dummy_opts, load_lut=False, train=False, db_root=DB_ROOT)
    grids_np = ds_tool.buildLookupTable(output_gpu_tensor=False)
    # Load grids lên device, không cần batch dimension ở đây vì Dataset sẽ trả về
    grids_tensor = [torch.from_numpy(g.astype(np.float32)).to(DEVICE) for g in grids_np]

    # 3. Prepare Data Loader
    print("3. Preparing Data Loader...")
    calib_ds = CalibrationDataset(IMG_ROOT_DIR, grids_tensor, limit=NUM_CALIB_SAMPLES)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False)

    # 4. Wrap & Fold BN
    print("4. Wrapping & Folding BN...")
    wrapper = ROmniStereoWrapper(model).to(DEVICE)
    wrapper.eval()
    
    # Dummy input phải khớp với signature mới của forward: forward(inputs_tuple)
    # Inputs tuple chứa 6 tensor.
    dummy_img = torch.randn(1, 1, 768, 800).to(DEVICE)
    dummy_grid = grids_tensor[0] # (H, W, D, 2)
    
    # Inputs thực sự là 1 tuple chứa 6 món
    real_input_tuple = (dummy_img, dummy_img, dummy_img, dummy_grid, dummy_grid, dummy_grid)
    
    # Dummy input cho AIMET (để gọi wrapper(*dummy_input)) phải là tuple chứa tuple trên
    dummy_input_for_aimet = (real_input_tuple, )

    # Dùng dummy_input thay vì input_shapes để BN fold tự trace
    _ = fold_all_batch_norms(wrapper, dummy_input=dummy_input_for_aimet)

    # =========================================================================
    # 5. APPLY ADAROUND
    # =========================================================================
    print("5. Applying AdaRound (Adaptive Rounding)...")
    
    params = AdaroundParameters(
        data_loader=calib_loader,
        num_batches=len(calib_loader),
        default_num_iterations=10000, 
        default_reg_param=0.01,
        default_beta_range=(20, 2),
        default_warm_start=0.2
    )

    adarounded_model = Adaround.apply_adaround(
        wrapper, 
        dummy_input_for_aimet, 
        params,
        path=OUTPUT_DIR,
        filename_prefix='adaround',
        default_param_bw=8,
        default_quant_scheme=QuantScheme.post_training_tf
    )

    print("✅ AdaRound Complete!")

    # =========================================================================
    # 6. CREATE QUANTSIM
    # =========================================================================
    print("6. Creating QuantSim using AdaRounded Model...")
    
    sim = QuantizationSimModel(
        model=adarounded_model,
        dummy_input=dummy_input_for_aimet,
        quant_scheme=QuantScheme.post_training_tf,
        default_output_bw=8,
        default_param_bw=8
    )

    # =========================================================================
    # 7. COMPUTE ACTIVATION ENCODINGS
    # =========================================================================
    print("7. Computing Activation Encodings...")
    sim.compute_encodings(forward_pass_callback=calibration_callback, forward_pass_callback_args=calib_loader)

    # =========================================================================
    # 8. EXPORT
    # =========================================================================
    print(f"8. Exporting to {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # Dọn dẹp
    del calib_loader, calib_ds, ds_tool
    import gc; gc.collect()

    try:
        sim.onnx.export(
            output_dir=OUTPUT_DIR,
            filename_prefix="romni_adarounded",
            dummy_input=dummy_input_for_aimet,
            opset_version=11
        )
        print("\n✅ DONE! Exported using sim.onnx.export (Opset 11)")
        
    except (TypeError, AttributeError):
        print("⚠️ Warning: sim.onnx.export failed, falling back to sim.export...")
        sim.export(
            path=OUTPUT_DIR,
            filename_prefix="romni_adarounded",
            dummy_input=dummy_input_for_aimet
        )
        print("\n✅ DONE! Exported using sim.export")

    print(f"  - {os.path.join(OUTPUT_DIR, 'romni_adarounded.onnx')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'romni_adarounded.encodings')}")

if __name__ == "__main__":
    main()