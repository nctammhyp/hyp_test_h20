import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import os
from easydict import EasyDict as Edict
from dataset import Dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
ONNX_PATH = r"F:\algo\mvs_v119\checkpoints\onnx\romnistereo32_v20_bs8_e46_jetson.onnx"
DB_ROOT = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable"
DB_NAME = "omnithings"

VIDEO_PATHS = [
    r"F:\Full-Dataset\hyp_data\hyp_data_01\FisheyeDatasetDepth\video\Pos1_Fisheye_CameraUB_2_1.mp4",
    r"F:\Full-Dataset\hyp_data\hyp_data_01\FisheyeDatasetDepth\video\Pos1_Fisheye_CameraUFL_2_2.mp4",
    r"F:\Full-Dataset\hyp_data\hyp_data_01\FisheyeDatasetDepth\video\Pos1_Fisheye_CameraUFR_2_3.mp4"
]

OUTPUT_DIR = "video_inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. INFERENCE ENGINE CLASS
# =============================================================================
class DepthInference:
    def __init__(self, onnx_path, db_root, db_name):
        print(f"\n--- [INIT] System Startup ---")
        self.input_size = (400, 384) # (W, H)
        
        # Load Calibration
        inference_opts = Edict()
        inference_opts.use_rgb = False
        inference_opts.num_downsample = 1
        
        print(f" -> Loading Dataset: {db_name}")
        self.dataset_tool = Dataset(db_name, db_opts=inference_opts, load_lut=False, train=False, db_root=db_root)
        
        print(f" -> Building Grids...")
        self.grids = self.dataset_tool.buildLookupTable(output_gpu_tensor=False)
        self.grids_onnx = [g.astype(np.float32) for g in self.grids]
        
        # Load ONNX
        print(f" -> Loading ONNX Model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.output_names = [node.name for node in self.session.get_outputs()]
        self.masks = [cam.invalid_mask for cam in self.dataset_tool.ocams]
        
        print(f"✅ System Ready. Device: {self.session.get_providers()[0]}")

    def preprocess_frame(self, frame, cam_idx):
        """Xử lý frame từ video (giống preprocess ảnh)"""
        if frame is None:
            return np.zeros((1, 1, self.input_size[1], self.input_size[0]), dtype=np.float32)

        # Chuyển xám nếu model yêu cầu grayscale
        if len(frame.shape) == 3:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            img = frame

        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        mask = self.masks[cam_idx]
        if mask is not None and mask.shape != (self.input_size[1], self.input_size[0]):
             mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32)
        valid_pixels = img[mask == 0] if mask is not None else img
        mean = np.mean(valid_pixels)
        std = np.std(valid_pixels) + 1e-6
        img = (img - mean) / std
        
        if mask is not None:
            img[mask > 0] = 0

        return img[np.newaxis, np.newaxis, :, :].astype(np.float32)

    def run_inference(self, frames):
        """Chạy model trên 3 frames và trả về cả InvDepth để tính mét"""
        imgs = [self.preprocess_frame(frames[i], i) for i in range(3)]
        
        input_feed = {
            "img0": imgs[0], "img1": imgs[1], "img2": imgs[2],
            "grid0": self.grids_onnx[0],
            "grid1": self.grids_onnx[1],
            "grid2": self.grids_onnx[2]
        }
        
        outputs = self.session.run(self.output_names, input_feed)
        pred_idx = np.squeeze(outputs[0]) 
        
        # Lấy Inverse Depth để tính toán khoảng cách thực (mét)
        inv_depth_map = self.dataset_tool.indexToInvdepth(pred_idx)
        # Lấy GT Format (0-255) theo logic của file .pt
        gt_formatted_map = self.dataset_tool.invdepth2gt(inv_depth_map)
        
        return inv_depth_map, gt_formatted_map

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
def main():
    # Khởi tạo Engine
    engine = DepthInference(ONNX_PATH, DB_ROOT, DB_NAME)
    
    # Mở 3 video
    caps = [cv2.VideoCapture(p) for p in VIDEO_PATHS]
    
    # Kiểm tra
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {VIDEO_PATHS[i]}")
            return

    frame_count = 0
    print("-> Starting Video Inference with Layout: 3 Top / 1 Bottom...")

    try:
        while True:
            rets = [cap.read() for cap in caps]
            if not all([r[0] for r in rets]):
                print("-> End of video stream.")
                break
            frames = [r[1] for r in rets]

            # 1. Inference
            _, gt_map = engine.run_inference(frames)

            # 2. Visualization - Layout: 3 Trên, 1 Dưới (Span)
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 3) 

            # --- Hàng trên: 3 Camera ---
            # Cam 1 (Trái)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
            ax1.set_title("Cam 1")
            ax1.axis('off')

            # Cam 2 (Giữa)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB))
            ax2.set_title("Cam 2")
            ax2.axis('off')

            # Cam 3 (Phải)
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(cv2.cvtColor(frames[2], cv2.COLOR_BGR2RGB))
            ax3.set_title("Cam 3")
            ax3.axis('off')

            # --- Hàng dưới: Depth Map ---
            # Span toàn bộ cột của hàng dưới (gs[1, :])
            ax4 = fig.add_subplot(gs[1, :]) 
            
            # Vẽ Depth với Jet Cmap
            ax4.imshow(gt_map, cmap='jet', aspect='equal') 
            ax4.set_title("Predicted Depth Output (Jet Cmap)")
            ax4.axis('off')

            # 3. Lưu ảnh
            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count:05d}.png")
            plt.savefig(save_path, dpi=100)
            plt.close(fig) # Giải phóng RAM

            if frame_count % 10 == 0:
                print(f"Processed frame {frame_count}...")

            frame_count += 1

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for cap in caps: cap.release()
        print(f"✅ Done! Results saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()