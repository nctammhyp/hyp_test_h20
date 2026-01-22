import os
import numpy as np
import argparse
from tqdm import tqdm
from easydict import EasyDict as Edict

# Import dataset của dự án
from dataset import Dataset

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Đường dẫn đến thư mục dataset gốc
DB_ROOT = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable"
DB_NAME = "omnithings"

# Thư mục sẽ chứa các file .raw
OUTPUT_DIR = "qnn_raw_data"
INPUT_LIST_FILE = "input_list.txt"

# Số lượng mẫu dùng để Quantize (50-100 là chuẩn)
NUM_SAMPLES = 500 

# =============================================================================
# 2. GENERATOR SCRIPT
# =============================================================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print("--- 1. Initializing Dataset ---")
    # Config giả để init dataset, đảm bảo logic giống hệt lúc train
    opts = Edict()
    opts.use_rgb = False
    opts.num_downsample = 1
    
    # Init dataset (load_lut=False để tự tính grid mới nhất)
    ds = Dataset(DB_NAME, db_opts=opts, load_lut=False, train=False, db_root=DB_ROOT)
    
    print(f"Dataset loaded. Total test samples available: {len(ds.test_idx)}")
    limit = min(NUM_SAMPLES, len(ds.test_idx))
    print(f"Generating data for {limit} samples...")

    # --- A. PREPARE GRIDS (STATIC INPUTS) ---
    print("--- 2. Saving Grid Data (Static) ---")
    # Grid chỉ cần tính 1 lần và dùng chung cho mọi ảnh
    # Output: List 3 numpy arrays [H, W, D, 2]
    grids = ds.buildLookupTable(output_gpu_tensor=False)
    
    grid_paths = []
    for i in range(3):
        # Chuyển sang Float32 (Quan trọng cho QNN)
        g_data = grids[i].astype(np.float32)
        
        # Lưu file raw
        filename = f"grid_cam{i}.raw"
        file_path = os.path.join(OUTPUT_DIR, filename)
        g_data.tofile(file_path)
        
        grid_paths.append(file_path)
        print(f"   Saved: {file_path} | Shape: {g_data.shape}")

    # --- B. PREPARE IMAGES (DYNAMIC INPUTS) ---
    print("--- 3. Saving Image Data (Dynamic) ---")
    list_lines = []

    for i in tqdm(range(limit)):
        fidx = ds.test_idx[i]
        
        # Hàm loadSample đã thực hiện: Read -> Resize -> Normalize -> Transpose (CHW)
        # Output imgs là list 3 numpy array [C, H, W]
        imgs, _, _, _ = ds.loadSample(fidx)
        
        current_line_paths = []
        
        # Lưu 3 ảnh (img0, img1, img2)
        for cam_idx in range(3):
            img_data = imgs[cam_idx].astype(np.float32)
            
            # Thêm batch dimension [1, C, H, W] để khớp với input model 1,1,768,800
            # loadSample trả về [C, H, W], ta cần [1, C, H, W]
            img_data = img_data[np.newaxis, ...]
            
            filename = f"img_cam{cam_idx}_frame{fidx}.raw"
            file_path = os.path.join(OUTPUT_DIR, filename)
            img_data.tofile(file_path)
            
            current_line_paths.append(file_path)

        # Thêm 3 đường dẫn Grid vào sau 3 đường dẫn Ảnh
        # Thứ tự input model: img0 img1 img2 grid0 grid1 grid2
        current_line_paths.extend(grid_paths)
        
        # Nối thành 1 dòng string
        line_str = " ".join(current_line_paths)
        list_lines.append(line_str)

    # --- C. WRITE INPUT LIST FILE ---
    print(f"--- 4. Writing {INPUT_LIST_FILE} ---")
    with open(INPUT_LIST_FILE, "w") as f:
        for line in list_lines:
            f.write(line + "\n")

    print("✅ DONE! Data preparation complete.")
    print(f"Run qairt-quantizer with '--input_list {INPUT_LIST_FILE}'")

if __name__ == "__main__":
    main()