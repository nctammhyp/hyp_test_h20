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
OUTPUT_DIR = "video_inference_results_depth_only"
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
        if frame is None:
            return np.zeros((1, 1, self.input_size[1], self.input_size[0]), dtype=np.float32)

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
        imgs = [self.preprocess_frame(frames[i], i) for i in range(3)]
        
        input_feed = {
            "img0": imgs[0], "img1": imgs[1], "img2": imgs[2],
            "grid0": self.grids_onnx[0],
            "grid1": self.grids_onnx[1],
            "grid2": self.grids_onnx[2]
        }
        
        outputs = self.session.run(self.output_names, input_feed)
        pred_idx = np.squeeze(outputs[0]) 
        
        inv_depth_map = self.dataset_tool.indexToInvdepth(pred_idx)
        gt_formatted_map = self.dataset_tool.invdepth2gt(inv_depth_map)
        
        return inv_depth_map, gt_formatted_map

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
def main():
    engine = DepthInference(ONNX_PATH, DB_ROOT, DB_NAME)
    
    caps = [cv2.VideoCapture(p) for p in VIDEO_PATHS]
    
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {VIDEO_PATHS[i]}")
            return

    frame_count = 0
    print("-> Starting Video Inference. Saving ONLY depth frames...")

    try:
        while True:
            rets = [cap.read() for cap in caps]
            if not all([r[0] for r in rets]):
                print("-> End of video stream.")
                break
            frames = [r[1] for r in rets]

            # 1. Inference
            _, gt_map = engine.run_inference(frames)

            # 2. Save ONLY Depth Image
            save_path = os.path.join(OUTPUT_DIR, f"depth_{frame_count:05d}.png")
            
            # Hàm imsave tự động áp dụng colormap 'jet' và lưu file ảnh sạch (không có viền/axis)
            plt.imsave(save_path, gt_map, cmap='jet')

            if frame_count % 10 == 0:
                print(f"Saved depth frame {frame_count}...")

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