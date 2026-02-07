import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import os
from easydict import EasyDict as Edict
from dataset import Dataset
from ocamcamera import OcamCamera

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

OCAM_FILE = r"F:\algo\mvs_v119\ocam1_hyp.txt"
MASK_PATH = r"F:\algo\mvs_v119\mask.png"
FISHEYE_W, FISHEYE_H = 800, 768

OUTPUT_DIR = "combined_inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. INFERENCE ENGINE CLASS
# =============================================================================
class DepthInference:
    def __init__(self, onnx_path, db_root, db_name, ocam_path):
        print(f"\n--- [INIT] System Startup ---")
        self.input_size = (400, 384) 
        
        # 1. Load Dataset Tool
        inference_opts = Edict()
        inference_opts.use_rgb = False
        inference_opts.num_downsample = 1
        self.dataset_tool = Dataset(db_name, db_opts=inference_opts, load_lut=False, train=False, db_root=db_root)
        
        # 2. Load OCam Model
        print(f" -> Loading OCam model: {ocam_path}")
        self.ocam_model = OcamCamera(ocam_path, fov=360)
        
        # 3. Load ONNX Session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.output_names = [node.name for node in self.session.get_outputs()]
        
        # 4. Prepare Grids & Masks (của model đầu vào)
        self.grids_onnx = [g.astype(np.float32) for g in self.dataset_tool.buildLookupTable(output_gpu_tensor=False)]
        self.masks = [cam.invalid_mask for cam in self.dataset_tool.ocams]

        # 5. Initialize Map for Reversion
        # Chúng ta khởi tạo map = None và sẽ tạo nó dựa trên size thực tế của pano output ở frame đầu tiên
        self.map_x = None
        self.map_y = None
        
        print(f"✅ System Ready. Device: {self.session.get_providers()[0]}")

    def create_inverse_map(self, f_w, f_h, p_w, p_h):
        """Logic từ file test_infer_2.py của bạn"""
        print(f" -> Precomputing inverse map (Fisheye {f_w}x{f_h} -> Pano {p_w}x{p_h})...")
        xx, yy = np.meshgrid(np.arange(f_w), np.arange(f_h))
        points2D = np.stack([xx.flatten(), yy.flatten()], axis=0)

        rays = self.ocam_model.cam2world(points2D)
        X, Y, Z = rays
        norm = np.sqrt(X*X + Y*Y + Z*Z)
        
        theta = np.arcsin(Y / norm) 
        phi = np.arctan2(X, Z)

        u = (phi + np.pi) / (2 * np.pi) * p_w
        v = (theta + np.pi / 2) / np.pi * p_h

        mapx = u.reshape(f_h, f_w).astype(np.float32)
        mapy = v.reshape(f_h, f_w).astype(np.float32)
        return mapx, mapy

    def apply_mask(self, image, target_size):
        """Logic từ file test_infer_2.py của bạn"""
        w, h = target_size
        if not os.path.exists(MASK_PATH):
            return image

        mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return image

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
        return cv2.bitwise_and(image, image, mask=binary_mask)

    def preprocess_frame(self, frame, cam_idx):
        if frame is None:
            return np.zeros((1, 1, self.input_size[1], self.input_size[0]), dtype=np.float32)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        mask = self.masks[cam_idx]
        if mask is not None:
            mask_uint8 = mask.astype(np.uint8)
            mask_res = cv2.resize(mask_uint8, self.input_size, interpolation=cv2.INTER_NEAREST)
            valid_pixels = img[mask_res == 0]
            mean, std = np.mean(valid_pixels), np.std(valid_pixels) + 1e-6
            img = (img - mean) / std
            img[mask_res > 0] = 0
        return img[np.newaxis, np.newaxis, :, :].astype(np.float32)

    def run_inference(self, frames):
        # Forward Model
        imgs = [self.preprocess_frame(frames[i], i) for i in range(3)]
        input_feed = {f"img{i}": imgs[i] for i in range(3)}
        input_feed.update({f"grid{i}": self.grids_onnx[i] for i in range(3)})
        
        outputs = self.session.run(self.output_names, input_feed)
        pred_idx = np.squeeze(outputs[0]) 
        
        # Pano Output
        inv_depth_map = self.dataset_tool.indexToInvdepth(pred_idx)
        pano_depth = self.dataset_tool.invdepth2gt(inv_depth_map) 

        # Khởi tạo map dựa trên kích thước thực tế của pano_depth
        if self.map_x is None:
            p_h, p_w = pano_depth.shape[:2]
            self.map_x, self.map_y = self.create_inverse_map(FISHEYE_W, FISHEYE_H, p_w, p_h)

        # Revert Pano -> Fisheye
        fisheye_depth = cv2.remap(
            pano_depth, 
            self.map_x, 
            self.map_y, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Áp dụng Mask
        fisheye_depth = self.apply_mask(fisheye_depth, (FISHEYE_W, FISHEYE_H))
        
        return pano_depth, fisheye_depth

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
def main():
    engine = DepthInference(ONNX_PATH, DB_ROOT, DB_NAME, OCAM_FILE)
    caps = [cv2.VideoCapture(p) for p in VIDEO_PATHS]
    
    frame_count = 0
    try:
        while True:
            rets = [cap.read() for cap in caps]
            if not all([r[0] for r in rets]): break
            frames = [r[1] for r in rets]

            pano_depth, fisheye_depth = engine.run_inference(frames)

            # Visualization Layout
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 3) 

            # Hàng 1: 3 Ảnh Fisheye RGB
            for i in range(3):
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
                ax.set_title(f"Cam {i+1} Original RGB")
                ax.axis('off')

            # Hàng 2: Depth Comparison
            # Bên trái: Reverted Fisheye Depth
            ax_f = fig.add_subplot(gs[1, 0])
            ax_f.imshow(fisheye_depth, cmap='jet')
            ax_f.set_title(f"Reverted Fisheye Depth\n({FISHEYE_W}x{FISHEYE_H})")
            ax_f.axis('off')

            # Bên phải: Panorama Depth (Span 2 ô)
            ax_p = fig.add_subplot(gs[1, 1:]) 
            ax_p.imshow(pano_depth, cmap='jet', aspect='auto') 
            ax_p.set_title(f"Predicted Panorama Depth\n({pano_depth.shape[1]}x{pano_depth.shape[0]})")
            ax_p.axis('off')

            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count:05d}.png")
            plt.savefig(save_path, dpi=100)
            plt.close(fig) 

            if frame_count % 10 == 0:
                print(f"Processed frame {frame_count}...")
            frame_count += 1

    finally:
        for cap in caps: cap.release()
        print(f"✅ Done! Results in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()