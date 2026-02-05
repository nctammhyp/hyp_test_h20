import os
import numpy as np
import argparse
from tqdm import tqdm
from easydict import EasyDict as Edict

# Import Dataset từ code hiện tại của bạn để đảm bảo tính nhất quán
from dataset import Dataset

# ================= CẤU HÌNH (PHẢI KHỚP VỚI EXPORT_ONNX_JETSON.PY) =================
# 1. Cấu hình đường dẫn Dataset trên Windows
DB_ROOT = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable"  # <-- SỬA ĐƯỜNG DẪN CỦA BẠN
DB_NAME = "omnithings"

# 2. Cấu hình Output
OUTPUT_DIR = "calib_data_npy"
NUM_SAMPLES = 1000  # Số lượng mẫu dùng để Calibration (nên từ 100-200)

# 3. Cấu hình Model (Khớp hoàn toàn với export_onnx_jetson.py)
# Input ảnh
H_IN, W_IN = 384, 400
# Input Grid (equirect_size=[128, 400], num_downsample=1 => Grid size = 64x200)
EQUIRECT_SIZE = [128, 400] 
NUM_DOWNSAMPLE = 1
NUM_INVDEPTH = 48
USE_RGB = False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"--- Chuẩn bị dữ liệu Calibration tại: {OUTPUT_DIR} ---")

    # 1. KHỞI TẠO DATASET
    # Logic này lấy từ train_u.py và export_onnx.py
    print("-> Loading Dataset...")
    opts = Edict()
    opts.data_opts = Edict({
        'phi_deg': 45.0, 
        'num_invdepth': NUM_INVDEPTH, 
        'equirect_size': EQUIRECT_SIZE, 
        'num_downsample': NUM_DOWNSAMPLE, 
        'use_rgb': USE_RGB
    })
    
    # Init Dataset (load_lut=False để script tự tính lại grid cho chắc chắn)
    # train=False để lấy tập test/validation (tránh dùng ảnh đã train để calib nếu có thể)
    ds = Dataset(DB_NAME, db_opts=opts.data_opts, load_lut=False, train=False, db_root=DB_ROOT)

    # 2. TẠO & LƯU GRID (INPUT TĨNH)
    print("-> Generating Grids...")
    # buildLookupTable trả về list 3 numpy array [H, W, D, 2]
    grids_np = ds.buildLookupTable(output_gpu_tensor=False)
    
    # Kiểm tra shape grid xem có đúng ý đồ export không (64, 200, 64, 2)
    print(f"   Grid Shape: {grids_np[0].shape}") 
    
    # Lưu Grid (Thêm Batch dimension [1, H, W, D, 2] để TensorRT dễ hiểu)
    # Lưu ý: inputs ONNX của bạn là: img0, img1, img2, grid0, grid1, grid2
    grid_inputs = []
    for i in range(3):
        g = grids_np[i].astype(np.float32)
        g = np.expand_dims(g, axis=0) # [1, H, W, D, 2]
        grid_inputs.append(g)

    # 3. VÒNG LẶP LƯU ẢNH (INPUT ĐỘNG)
    # Lấy ngẫu nhiên hoặc tuần tự NUM_SAMPLES ảnh
    # indices = ds.test_idx[:NUM_SAMPLES]
    indices = ds.train_idx[:NUM_SAMPLES]
    if len(indices) < NUM_SAMPLES:
        indices = ds.test_idx # Lấy hết nếu không đủ
    
    print(f"-> Processing {len(indices)} samples...")
    
    for i, idx in enumerate(tqdm(indices)):
        # ds.loadSample tự động làm: Read -> Resize (theo code dataset) -> Normalize -> Transpose CHW
        # Chúng ta chỉ cần tin tưởng vào dataset.py
        imgs, _, _, _ = ds.loadSample(idx)
        
        # imgs là list 3 ảnh [C, H, W] (Float32, Normalized)
        
        # Kiểm tra kích thước xem dataset có resize đúng 384x400 không
        # Nếu dataset.py của bạn đang resize về 800x768 (như trong file dataset.py bạn up),
        # chúng ta CẦN resize lại về 384x400 ở đây để khớp với ONNX Jetson.
        
        processed_imgs = []
        for img in imgs:
            # img shape: [C, H, W] -> [1, H, W] (vì grayscale)
            # Nếu size chưa đúng 384x400 thì resize
            if img.shape[1] != H_IN or img.shape[2] != W_IN:
                # Transpose về [H, W, C] để cv2 resize
                img_hwc = np.transpose(img, (1, 2, 0)) 
                img_resized = cv2.resize(img_hwc, (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)
                
                # Trả về [C, H, W]
                if len(img_resized.shape) == 2:
                    img_resized = img_resized[np.newaxis, :, :]
                else:
                    img_resized = np.transpose(img_resized, (2, 0, 1))
                img = img_resized

            # Thêm Batch Dimension [1, C, H, W]
            img = np.expand_dims(img, axis=0)
            processed_imgs.append(img.astype(np.float32))

        # Lưu tất cả 6 input vào 1 file .npz để bên Jetson load cho gọn
        save_path = os.path.join(OUTPUT_DIR, f"batch_{i}.npz")
        
        np.savez(save_path, 
                 img0=processed_imgs[0], 
                 img1=processed_imgs[1], 
                 img2=processed_imgs[2], 
                 grid0=grid_inputs[0], 
                 grid1=grid_inputs[1], 
                 grid2=grid_inputs[2])

    print("✅ Hoàn tất!")
    print(f"Copy thư mục '{OUTPUT_DIR}' sang Jetson Orin Nano để chạy Calibration.")

if __name__ == "__main__":
    # Cần import cv2 nếu dataset resize chưa đúng
    import cv2 
    main()