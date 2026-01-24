import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from easydict import EasyDict as Edict

# Import từ các file của bạn
from dataset import Dataset as ProjectDataset
from aimet_quantize import CalibrationDataset

# Cấu hình đường dẫn
DB_ROOT = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable"
DB_NAME = "omnithings"
IMG_ROOT_DIR = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable\omnithings"
RAW_DATA_DIR = r"qnn_calib_raw" # Thư mục chứa các file .raw
INPUT_LIST_FILE = r"input_list.txt"
NUM_SAMPLES = 5 # Số lượng mẫu dùng để calib (nên khớp với NUM_CALIB_SAMPLES)

def main():
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    # 1. Load Grids (Lấy grid thực tế giống aimet_quantize.py)
    print("Loading Grids...")
    dummy_opts = Edict({'use_rgb': False, 'num_downsample': 1})
    ds_tool = ProjectDataset(DB_NAME, db_opts=dummy_opts, load_lut=False, train=False, db_root=DB_ROOT)
    grids_np = ds_tool.buildLookupTable(output_gpu_tensor=False)
    grids_tensor = [torch.from_numpy(g.astype(np.float32)) for g in grids_np]

    # 2. Khởi tạo DataLoader
    calib_ds = CalibrationDataset(IMG_ROOT_DIR, grids_tensor, limit=NUM_SAMPLES)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False)

    input_list_lines = []

    print(f"Generating {NUM_SAMPLES} samples...")
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            # batch chứa: img0, img1, img2, grid0, grid1, grid2
            img0, img1, img2, g0, g1, g2 = batch
            
            # Danh sách để mapping tên và tensor
            data_map = {
                "img0": img0, "img1": img1, "img2": img2,
                "grid0": g0, "grid1": g1, "grid2": g2
            }

            line_parts = []
            for name, tensor in data_map.items():
                # Chuyển về numpy float32 và lưu dạng binary (.raw)
                raw_filename = f"{name}_sample{i}.raw"
                raw_path = os.path.join(RAW_DATA_DIR, raw_filename)
                
                # Quan trọng: Qualcomm yêu cầu dữ liệu thô (C-style order)
                tensor.numpy().astype(np.float32).tofile(raw_path)
                
                # Tạo chuỗi "tên_input:=đường_dẫn"
                line_parts.append(f"{name}:={raw_path}")

            # Nối các phần tử của 6 input thành 1 dòng
            input_list_lines.append(" ".join(line_parts))

    # 3. Ghi vào file input_list.txt
    with open(INPUT_LIST_FILE, "w") as f:
        f.write("\n".join(input_list_lines))

    print(f"✅ Thành công! Đã tạo thư mục {RAW_DATA_DIR} và file {INPUT_LIST_FILE}")

if __name__ == "__main__":
    main()