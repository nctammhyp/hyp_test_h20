import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import os
import sys
from easydict import EasyDict as Edict # Import thêm Edict

# Import dataset
from dataset import Dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
ONNX_PATH = r"F:\algo\mvs_v119\romnistereo_v11_fixed.onnx"
DB_ROOT = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable"
DB_NAME = "omnithings"

# SỬA LẠI ĐƯỜNG DẪN ẢNH CỦA BẠN CHO ĐÚNG
IMG_PATHS = [
    r"F:\algo\mvs_v119\omnidata\hyp_01\cam1\00001.png",
    r"F:\algo\mvs_v119\omnidata\hyp_01\cam2\00001.png",
    r"F:\algo\mvs_v119\omnidata\hyp_01\cam3\00001.png"
]

# =============================================================================
# 2. INFERENCE ENGINE CLASS
# =============================================================================
class DepthInference:
    def __init__(self, onnx_path, db_root, db_name):
        print(f"\n--- [INIT] System Startup ---")
        
        # Kiểm tra đường dẫn
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model not found: {onnx_path}")
        if not os.path.exists(os.path.join(db_root, db_name)):
            raise FileNotFoundError(f"Database not found: {os.path.join(db_root, db_name)}")

        self.input_size = (800, 768) # Kích thước input resize
        
        # --- A. Load Calibration (Tái sử dụng class Dataset) ---
        # Tạo config giả để tránh lỗi 'use_rgb' trong dataset.py
        inference_opts = Edict()
        inference_opts.use_rgb = False
        inference_opts.num_downsample = 1
        
        print(f" -> Loading Calibration from: {db_name}")
        # load_lut=False để ta tự build lại grid bên dưới
        self.dataset_tool = Dataset(db_name, db_opts=inference_opts, load_lut=False, train=False, db_root=db_root)
        
        print(f" -> Computing Rectification Grids...")
        self.grids = self.dataset_tool.buildLookupTable(output_gpu_tensor=False)
        
        # Convert Grid sang Float32 cho ONNX
        # Lưu ý: Model ONNX nhận Grid Rank 4 [H, W, D, 2], KHÔNG thêm batch dimension
        self.grids_onnx = [g.astype(np.float32) for g in self.grids]

        # --- B. Load ONNX Model ---
        print(f" -> Loading ONNX Session...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        
        self.output_names = [node.name for node in self.session.get_outputs()]
        self.masks = [cam.invalid_mask for cam in self.dataset_tool.ocams]

        print(f"✅ System Ready. Device: {self.session.get_providers()[0]}")

    def preprocess(self, img_path, cam_idx):
        """ Đọc ảnh -> Resize -> Normalize -> Add Batch Dim """
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: Image not found {img_path}, using black image.")
            img = np.zeros((768, 800), dtype=np.uint8)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
        if img is None: raise ValueError(f"Bad image: {img_path}")

        # 1. Resize về 800x768
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # 2. Normalize (Giống logic dataset.py)
        mask = self.masks[cam_idx]
        if mask is not None and mask.shape != (768, 800):
             mask = cv2.resize(mask, (800, 768), interpolation=cv2.INTER_NEAREST)
             
        img = img.astype(np.float32)
        
        # Tính mean/std trên vùng valid
        valid_pixels = img[mask == 0] if mask is not None else img
        mean = np.mean(valid_pixels)
        std = np.std(valid_pixels) + 1e-6
        img = (img - mean) / std
        
        # Gán 0 cho vùng invalid
        if mask is not None:
            img[mask > 0] = 0

        # 3. Add Batch & Channel dims: [H, W] -> [1, 1, H, W]
        img = img[np.newaxis, np.newaxis, :, :] 
        return img.astype(np.float32)

    def run(self, img_paths):
        print(f" -> Processing 3 images...")
        
        # 1. Preprocess Images
        img0 = self.preprocess(img_paths[0], cam_idx=0)
        img1 = self.preprocess(img_paths[1], cam_idx=1)
        img2 = self.preprocess(img_paths[2], cam_idx=2)
        
        # 2. Prepare Inputs
        input_feed = {
            "img0": img0, "img1": img1, "img2": img2,
            "grid0": self.grids_onnx[0],
            "grid1": self.grids_onnx[1],
            "grid2": self.grids_onnx[2]
        }
        
        # 3. ONNX Inference
        outputs = self.session.run(self.output_names, input_feed)
        
        # Lấy output thô (Raw Index 0..191)
        # Tương đương: invdepth_idx = net(...)
        pred_idx = np.squeeze(outputs[0]) 

        # 4. Post-process (Dùng logic giống hệt file .pt)
        print(" -> Applying Post-processing (Matching .pt logic)...")
        
        # Bước 1: Index -> Inverse Depth (1/m)
        # Tương đương: invdepth = data.indexToInvdepth(invdepth_idx)
        inv_depth_map = self.dataset_tool.indexToInvdepth(pred_idx)
        
        # Bước 2: Inverse Depth -> Ground Truth Format (0-255 / Quantized)
        # Tương đương: invdepth = data.invdepth2gt(invdepth)
        # Output này dùng để so sánh với GT hoặc lưu file ảnh grayscale
        gt_formatted_map = self.dataset_tool.invdepth2gt(inv_depth_map)
        
        return inv_depth_map, gt_formatted_map

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
def main():
    try:
        # Khởi tạo Engine
        engine = DepthInference(ONNX_PATH, DB_ROOT, DB_NAME)
        
        # Chạy Inference
        # Trả về cả InvDepth (để tính mét vẽ màu) và GT Format (để lưu ảnh xám chuẩn)
        inv_depth_map, gt_formatted_map = engine.run(IMG_PATHS)
        
        # --- Visualization ---
        plt.figure(figsize=(12, 5))
        
        # 1. Vẽ Depth Map (Mét) - Nhìn đẹp, trực quan
        # Convert InvDepth -> Depth (m) để visualize màu
        depth_meters = 1.0 / (inv_depth_map + 1e-6)
        depth_meters[depth_meters > 20] = 20 # Clip ở 20m cho dễ nhìn
        
        plt.subplot(1, 2, 1)
        plt.imshow(depth_meters, cmap='magma')
        plt.colorbar(label='Depth (m)')
        plt.title('Depth Map (Meters)')
        plt.axis('off')

        # 2. Vẽ GT Formatted (Grayscale) - Giống file .pt output
        plt.subplot(1, 2, 2)
        plt.imshow(gt_formatted_map, cmap='gray')
        plt.colorbar(label='Quantized (0-255)')
        plt.title('Output like .pt (invdepth2gt)')
        plt.axis('off')

        save_file = "inference_final_result.png"
        plt.tight_layout()
        plt.savefig(save_file)
        print(f"\n✅ SUCCESS! Visualization saved to: {os.path.abspath(save_file)}")
        plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
