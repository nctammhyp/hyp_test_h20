import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from easydict import EasyDict as Edict
from dataset import Dataset # Import class Dataset của bạn

# ==========================================
# CẤU HÌNH (Sửa lại đường dẫn của bạn)
# ==========================================
DB_ROOT = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable"
DB_NAME = "omnithings"
OUTPUT_DIR = "calibration_data" # Tên folder sẽ tạo ra
NUM_SAMPLES = 5  # Số lượng ảnh mẫu để Quantization (50-100 là đủ)

# Kích thước Input
INPUT_H, INPUT_W = 768, 800

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class QuantDataGenerator:
    def __init__(self, db_root, db_name):
        print("--- Đang khởi tạo Dataset Tool ---")
        # Setup giống inference.py để lấy thông số Camera và Grid
        opts = Edict()
        opts.use_rgb = False
        opts.num_downsample = 1
        
        # Load dataset tool để lấy config camera (OCam)
        self.dataset = Dataset(db_name, db_opts=opts, load_lut=False, train=True, db_root=db_root)
        
        # Tạo Grids (Lookup Tables) - Chỉ cần tạo 1 lần vì nó cố định
        print("--- Đang tính toán Rectification Grids ---")
        self.grids = self.dataset.buildLookupTable(output_gpu_tensor=False)
        self.masks = [cam.invalid_mask for cam in self.dataset.ocams]

    def preprocess_image(self, img_path, cam_idx):
        """ Xử lý ảnh giống hệt inference.py: Resize -> Normalize Mean/Std """
        if not os.path.exists(img_path):
            print(f"LỖI: Không tìm thấy {img_path}")
            return np.zeros((1, 1, INPUT_H, INPUT_W), dtype=np.float32)

        # 1. Read
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Resize
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)

        # 3. Normalize (Quan trọng: Phải khớp logic training/inference)
        mask = self.masks[cam_idx]
        if mask is not None and mask.shape != (INPUT_H, INPUT_W):
             mask = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
        
        # Tính mean/std trên vùng valid pixels
        valid_pixels = img[mask == 0] if mask is not None else img
        mean = np.mean(valid_pixels)
        std = np.std(valid_pixels) + 1e-6
        img = (img - mean) / std
        
        # Gán 0 cho vùng invalid
        if mask is not None:
            img[mask > 0] = 0

        # 4. Add Batch & Channel dims: [H, W] -> [1, 1, H, W]
        img = img[np.newaxis, np.newaxis, :, :] 
        return img.astype(np.float32)

    def generate(self):
        # Tạo cấu trúc thư mục
        imgs_dir = os.path.join(OUTPUT_DIR, "imgs")
        grids_dir = os.path.join(OUTPUT_DIR, "grids")
        ensure_dir(imgs_dir)
        ensure_dir(grids_dir)

        # 1. Lưu Grid Files (Dạng .raw float32)
        # Grid input shape: (80, 320, 96, 2) hoặc tương tự, tùy export. 
        # Ta save raw flat buffer, SNPE sẽ tự reshape theo DLC.
        print("--- Đang lưu file Grid .raw ---")
        grid_paths = []
        for i in range(3):
            grid_filename = f"grid{i}.raw"
            save_path = os.path.join(grids_dir, grid_filename)
            # Quan trọng: convert sang float32 trước khi lưu
            self.grids[i].astype(np.float32).tofile(save_path)
            # Lưu đường dẫn tương đối để dùng trên Linux
            grid_paths.append(os.path.join("grids", grid_filename))

        # 2. Random chọn ảnh và xử lý
        print(f"--- Đang xử lý {NUM_SAMPLES} mẫu ảnh ---")
        
        # Lấy danh sách index hợp lệ từ dataset
        all_indices = self.dataset.train_idx
        selected_indices = random.sample(all_indices, min(NUM_SAMPLES, len(all_indices)))
        
        input_list_path = os.path.join(OUTPUT_DIR, "input_list.txt")
        
        with open(input_list_path, "w") as f:
            # Ghi chú: Định dạng của SNPE input list cho multi-input:
            # InputName1:=Path1 InputName2:=Path2 ...
            
            for idx in tqdm(selected_indices):
                # Lấy đường dẫn ảnh gốc từ dataset
                # loadImages trả về list ảnh đã processed, nhưng ở đây ta cần path raw để tự process lại kiểm soát
                # Nên ta tự construct path dựa trên dataset config
                img_files = []
                for cam_i in range(3):
                    # Format: camX/00001.png (theo dataset.py)
                    rel_path = self.dataset.img_fmt % (cam_i + 1, idx)
                    full_path = os.path.join(self.dataset.db_path, rel_path)
                    
                    # Xử lý ảnh
                    img_tensor = self.preprocess_image(full_path, cam_i)
                    
                    # Lưu file .raw
                    raw_filename = f"img{cam_i}_frame{idx}.raw"
                    save_path = os.path.join(imgs_dir, raw_filename)
                    img_tensor.tofile(save_path)
                    
                    # Đường dẫn tương đối cho file txt
                    img_files.append(os.path.join("imgs", raw_filename))
                
                # Ghi 1 dòng vào input_list.txt
                # Định dạng: tên_input:=đường_dẫn
                # Tên input phải khớp với file ONNX/DLC: img0, img1, img2, grid0, grid1, grid2
                line_parts = []
                # Add Images
                line_parts.append(f"img0:={img_files[0]}")
                line_parts.append(f"img1:={img_files[1]}")
                line_parts.append(f"img2:={img_files[2]}")
                # Add Grids (Grid dùng chung cho tất cả các frame)
                line_parts.append(f"grid0:={grid_paths[0]}")
                line_parts.append(f"grid1:={grid_paths[1]}")
                line_parts.append(f"grid2:={grid_paths[2]}")
                
                f.write(" ".join(line_parts) + "\n")
                
        print(f"\n✅ Hoàn tất! Data nằm trong thư mục: {os.path.abspath(OUTPUT_DIR)}")
        print("Hãy copy thư mục này sang Ubuntu để chạy quantize.")

if __name__ == "__main__":
    gen = QuantDataGenerator(DB_ROOT, DB_NAME)
    gen.generate()