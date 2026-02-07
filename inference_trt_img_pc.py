import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Tự động khởi tạo CUDA context
import matplotlib.pyplot as plt
import os
import sys
from easydict import EasyDict as Edict

# Import dataset từ code của bạn
from dataset import Dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Đường dẫn file engine bạn vừa build xong
ENGINE_PATH = "romnistereo_v20_final_pc.engine" 

# Đường dẫn dataset (Sửa lại cho đúng trên Jetson)
DB_ROOT = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable"
DB_NAME = "omnithings"

# SỬA LẠI ĐƯỜNG DẪN ẢNH CỦA BẠN CHO ĐÚNG
IMG_PATHS = [
    r"F:\algo\mvs_v119\omnidata\hyp_02\cam1\00001.png",
    r"F:\algo\mvs_v119\omnidata\hyp_02\cam2\00001.png",
    r"F:\algo\mvs_v119\omnidata\hyp_02\cam3\00001.png"
]

# Kích thước input (Phải khớp với lúc build engine)
INPUT_H, INPUT_W = 384, 400

# Logger cho TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =============================================================================
# 2. TENSORRT WRAPPER CLASS
# =============================================================================
class TRTInference:
    def __init__(self, engine_path, db_root, db_name):
        print(f"\n--- [INIT] TensorRT System Startup ---")
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        # --- A. Load Dataset Tool (để lấy Grid & OCam) ---
        print(f" -> Loading Calibration/Grid info from: {db_name}")
        inference_opts = Edict()
        inference_opts.use_rgb = False
        inference_opts.num_downsample = 1
        # Set cứng size để dataset tính grid đúng với model
        inference_opts.equirect_size = [128, 400] 
        inference_opts.num_invdepth = 48
        
        self.dataset_tool = Dataset(db_name, db_opts=inference_opts, load_lut=False, train=False, db_root=db_root)
        
        # Tính toán Grid
        print(f" -> Computing Rectification Grids...")
        self.grids = self.dataset_tool.buildLookupTable(output_gpu_tensor=False)
        self.masks = [cam.invalid_mask for cam in self.dataset_tool.ocams]

        # --- B. Load TensorRT Engine ---
        print(f" -> Loading TensorRT Engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if not self.engine:
            raise RuntimeError("Failed to load TensorRT Engine.")

        self.context = self.engine.create_execution_context()
        
        # --- C. Allocate Memory (Host & Device) ---
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        print(" -> Allocating CUDA Memory...")
        
        # Duyệt qua các bindings (input/output) của engine
        # Lưu ý: TRT 10 API có thể khác, đây là cách tương thích ngược
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = self.engine.get_tensor_dtype(tensor_name)
            
            # Xử lý Dynamic Shape (-1) nếu có (thường đặt max batch size = 1)
            # Với model của bạn đã fix shape lúc build nên chắc không sao
            curr_shape = list(shape)
            if curr_shape[0] == -1: curr_shape[0] = 1 
            
            # Tính kích thước bộ nhớ cần cấp phát
            size = trt.volume(curr_shape)
            # Map TRT datatype sang Numpy datatype
            if dtype == trt.float32: np_dtype = np.float32
            elif dtype == trt.float16: np_dtype = np.float16
            elif dtype == trt.int8: np_dtype = np.int8
            elif dtype == trt.int32: np_dtype = np.int32
            else: np_dtype = np.float32

            # Cấp phát bộ nhớ
            host_mem = cuda.pagelocked_empty(size, np_dtype) # CPU (pinned memory)
            device_mem = cuda.mem_alloc(host_mem.nbytes)     # GPU
            
            # Lưu thông tin binding
            binding = {
                "index": i,
                "name": tensor_name,
                "shape": curr_shape,
                "dtype": np_dtype,
                "host": host_mem,
                "device": device_mem
            }
            self.bindings.append(int(device_mem)) # Danh sách địa chỉ GPU để execute

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
                print(f"    [Input]  {tensor_name}: {curr_shape} ({np_dtype})")
            else:
                self.outputs.append(binding)
                print(f"    [Output] {tensor_name}: {curr_shape} ({np_dtype})")

        print(f"✅ System Ready.")

    def preprocess(self, img_path, cam_idx):
        """ Read -> Resize (384x400) -> Normalize -> Add Batch Dim """
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: Image not found {img_path}, using black image.")
            img = np.zeros((INPUT_H, INPUT_W), dtype=np.uint8)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
        if img is None: raise ValueError(f"Bad image: {img_path}")

        # 1. Resize
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)

        # 2. Normalize (Khớp dataset.py)
        mask = self.masks[cam_idx]
        # Resize mask nếu cần
        if mask is not None and mask.shape != (INPUT_H, INPUT_W):
             mask = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)

        valid_pixels = img[mask == 0] if mask is not None else img
        mean = np.mean(valid_pixels)
        std = np.std(valid_pixels) + 1e-6
        img = (img - mean) / std
        
        if mask is not None: img[mask > 0] = 0

        # 3. Add Batch & Channel dims: [1, 1, H, W]
        img = img[np.newaxis, np.newaxis, :, :] 
        return np.ascontiguousarray(img) # Quan trọng cho CUDA copy

    def run(self, img_paths):
        print(f" -> Processing 3 images...")
        
        # 1. Preprocess Images
        img0 = self.preprocess(img_paths[0], cam_idx=0)
        img1 = self.preprocess(img_paths[1], cam_idx=1)
        img2 = self.preprocess(img_paths[2], cam_idx=2)
        
        # 2. Prepare Inputs Data Dictionary
        # Mapping tên input với dữ liệu numpy
        # Lưu ý: Shape của Grid phải khớp với binding (có thể cần add batch dim)
        
        input_data = {
            "img0": img0,
            "img1": img1,
            "img2": img2,
        }
        
        # Prepare Grids
        # Grid từ dataset là (H, W, D, 2), cần thêm batch dim -> (1, H, W, D, 2)
        for i, g_name in enumerate(["grid0", "grid1", "grid2"]):
            g = self.grids[i].astype(np.float32)
            g = np.expand_dims(g, axis=0) # [1, H, W, D, 2]
            input_data[g_name] = np.ascontiguousarray(g)

        # 3. Copy Data from Host to Device
        for inp in self.inputs:
            name = inp["name"]
            if name in input_data:
                # Copy numpy data vào vùng nhớ Host Pinned
                np.copyto(inp["host"], input_data[name].ravel())
                # Async Copy từ Host -> Device
                cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
            else:
                print(f"⚠️ Warning: Missing input data for {name}")

        # 4. Execute Inference
        # set_tensor_address là bắt buộc cho TRT 10 nếu dùng execute_v2 không nhận list bindings
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 5. Copy Data from Device to Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        
        # Đồng bộ hóa stream (đợi GPU chạy xong)
        self.stream.synchronize()

        # 6. Get Result
        # Giả sử output đầu tiên là depth map
        output_buffer = self.outputs[0]["host"]
        output_shape = self.outputs[0]["shape"] # [1, 1, 384, 400]
        
        # Reshape về [H, W]
        pred_idx = output_buffer.reshape(output_shape).squeeze()
        
        # 7. Post-process
        print(" -> Post-processing...")
        inv_depth_map = self.dataset_tool.indexToInvdepth(pred_idx)
        gt_formatted_map = self.dataset_tool.invdepth2gt(inv_depth_map)
        
        return inv_depth_map, gt_formatted_map

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================
def main():
    try:
        # Khởi tạo Engine
        engine = TRTInference(ENGINE_PATH, DB_ROOT, DB_NAME)
        
        # Chạy Inference
        inv_depth_map, gt_formatted_map = engine.run(IMG_PATHS)
        
        # --- Visualization ---
        plt.figure(figsize=(12, 5))
        
        # 1. Vẽ Depth Map (Mét)
        depth_meters = 1.0 / (inv_depth_map + 1e-6)
        depth_meters[depth_meters > 20] = 20
        
        plt.subplot(1, 2, 1)
        plt.imshow(depth_meters, cmap='magma')
        plt.colorbar(label='Depth (m)')
        plt.title('TensorRT INT8 Inference')
        plt.axis('off')

        # 2. Vẽ GT Formatted
        plt.subplot(1, 2, 2)
        plt.imshow(gt_formatted_map, cmap='gray')
        plt.colorbar(label='Quantized (0-255)')
        plt.title('Output (Gray)')
        plt.axis('off')

        save_file = "trt_inference_result.png"
        plt.tight_layout()
        plt.savefig(save_file)
        print(f"\n✅ SUCCESS! Saved result to: {os.path.abspath(save_file)}")
        plt.show() # Uncomment nếu chạy có màn hình

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()