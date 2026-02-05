import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import glob
import sys
import gc

# ================= CẤU HÌNH (SỬA ĐƯỜNG DẪN NẾU CẦN) =================
# Lưu ý: Hãy chắc chắn bạn đang dùng file ONNX được export với Opset 13
ONNX_PATH = "checkpoints/onnx/romnistereo32_v20_bs8_e46_jetson.onnx"
ENGINE_PATH = "checkpoints/onnx/romnistereo32_v20_bs8_e46_jetson_pc.engine"
CALIB_DATA_DIR = "calib_data_npy"
CACHE_FILE = "calib.cache"
BATCH_SIZE = 1

# GIỚI HẠN BỘ NHỚ TẠM (Workspace)
# Để 256MB (256 * 1024 * 1024) là an toàn nhất cho Jetson Orin Nano khi build model nặng
WORKSPACE_SIZE = 256 * 1024 * 1024 

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# ================= CLASS CALIBRATOR =================
class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, cache_file, input_shapes):
        super().__init__()
        self.cache_file = cache_file
        self.data_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.batch_size = BATCH_SIZE
        self.current_idx = 0
        self.input_shapes = input_shapes 
        self.device_buffers = {}
        
        if len(self.data_files) == 0:
            print(f"❌ Lỗi: Không tìm thấy file .npz trong '{data_dir}'")
            print("   -> Hãy chạy 'prepare_calib_data.py' trên Windows và copy thư mục sang.")
            sys.exit(1)
        print(f"-> Tìm thấy {len(self.data_files)} mẫu calibration.")

        # Cấp phát bộ nhớ GPU
        for name, size in self.input_shapes.items():
            try:
                self.device_buffers[name] = cuda.mem_alloc(size)
            except Exception as e:
                print(f"❌ OOM khi cấp phát input '{name}': {e}")
                sys.exit(1)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_idx >= len(self.data_files):
            return None

        # Dọn dẹp bộ nhớ Python định kỳ để tránh rò rỉ RAM
        if self.current_idx % 5 == 0:
            gc.collect()

        file_path = self.data_files[self.current_idx]
        if self.current_idx % 10 == 0:
            print(f"[Calib] Processing {self.current_idx + 1}/{len(self.data_files)}")
        
        try:
            data = np.load(file_path)
            ptrs = []
            for name in names:
                if name not in self.device_buffers: 
                    continue
                
                # TensorRT có thể tự đổi tên input trong quá trình optimize
                # Code này cố gắng map đúng tên
                if name not in data:
                    print(f"❌ Thiếu input '{name}' trong file npz. Các key có sẵn: {list(data.keys())}")
                    return None
                
                # Flatten dữ liệu và copy vào GPU
                arr = np.ascontiguousarray(data[name].ravel())
                cuda.memcpy_htod(self.device_buffers[name], arr)
                ptrs.append(int(self.device_buffers[name]))
            
            self.current_idx += 1
            return ptrs
        except Exception as e:
            print(f"❌ Lỗi đọc batch: {e}")
            return None

    def read_calibration_cache(self):
        # Nếu đã có cache, load lên để đỡ phải chạy lại
        if os.path.exists(self.cache_file):
            print(f"-> Loading cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f: return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"-> Writing cache to: {self.cache_file}")
        with open(self.cache_file, "wb") as f: f.write(cache)

# ================= BUILD ENGINE =================
def build_engine():
    # Xóa cache RAM hệ thống trước khi chạy
    os.system("sync; echo 3 > /proc/sys/vm/drop_caches")
    gc.collect()
    
    print(f"--- BẮT ĐẦU BUILD (Workspace Limit: {WORKSPACE_SIZE/1024**2:.0f} MB) ---")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 1. Parse ONNX (Dùng parse_from_file để fix lỗi file .data)
    if not os.path.exists(ONNX_PATH):
        print(f"❌ Không tìm thấy file ONNX: {ONNX_PATH}")
        return

    print("-> Parsing ONNX...")
    if not parser.parse_from_file(ONNX_PATH):
        print('❌ Lỗi Parse ONNX:')
        for error in range(parser.num_errors): print(parser.get_error(error))
        return

    # 2. Lấy thông tin Input
    input_shapes = {}
    print("-> Detected Inputs:")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        vol = 1
        for d in t.shape: vol *= max(1, d) # Handle dynamic dim (-1) -> 1
        input_shapes[t.name] = vol * 4 # float32
        print(f"   - {t.name}: {t.shape}")
    
    # 3. Cấu hình Tối ưu Bộ nhớ (QUAN TRỌNG)
    # Giới hạn Workspace thật chặt
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE)
    
    # Tắt Timing Cache để không tốn RAM lưu profile các layer
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    
    # Tắt bớt các tactics tốn kém của cuBLAS/cuDNN nếu cần thiết (Optional)
    # tactic_sources = 1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT)
    # config.set_tactic_sources(tactic_sources)

    # 4. Config INT8
    if builder.platform_has_fast_int8:
        print("✅ Chế độ INT8 Enabled.")
        config.set_flag(trt.BuilderFlag.INT8)
        # Cho phép fallback về FP16 ở các lớp nhạy cảm
        config.set_flag(trt.BuilderFlag.FP16)
        
        calibrator = INT8Calibrator(CALIB_DATA_DIR, CACHE_FILE, input_shapes)
        config.int8_calibrator = calibrator
    else:
        print("⚠️ Cảnh báo: Phần cứng không hỗ trợ INT8 thuần, chuyển sang FP16.")
        config.set_flag(trt.BuilderFlag.FP16)

    # # 5. Build Engine
    # print("-> Building Engine... (Quá trình này sẽ RẤT LÂU và dùng SWAP)")
    # print("-> Đừng lo nếu thấy RAM đỏ 100%, Swap sẽ gánh.")
    
    # try:
    #     # build_serialized_network trả về bytes
    #     engine_bytes = builder.build_serialized_network(network, config)
        
    #     if engine_bytes:
    #         with open(ENGINE_PATH, "wb") as f: f.write(engine_bytes)
    #         print(f"\n✅ SUCCESS! Engine lưu tại: {ENGINE_PATH}")
    #         print("Copy file này vào project để chạy inference.")
    #     else:
    #         print("\n❌ Build thất bại (Engine = None).")
    #         print("Nguyên nhân có thể do Opset ONNX quá cao (hãy dùng Opset 13) hoặc Swap chưa đủ.")

    # except Exception as e:
    #     print(f"\n❌ Build Crash: {e}")

    # 5. Build Engine
    print("-> Building Engine... (Quá trình này sẽ RẤT LÂU và dùng SWAP)")
    
    # === SỬA TỪ ĐÂY ===
    # Thay vì build toàn bộ, ta chỉ kích hoạt builder để nó chạy calibration rồi thoát
    try:
        # Cách để ép TensorRT chạy calibration mà không cần build full engine (tiết kiệm thời gian)
        # Tuy nhiên, API cũ không hỗ trợ "calib only" trực tiếp. 
        # Chúng ta cứ để nó build, NHƯNG quan trọng là file calib.cache được ghi ra.
        
        print("-> Đang chạy Calibration... (Hãy đợi đến khi thấy file 'calib.cache' xuất hiện)")
        engine_bytes = builder.build_serialized_network(network, config)
        
        # Ngay sau khi dòng này chạy xong, file calib.cache đã được tạo bởi class INT8Calibrator
        if os.path.exists(CACHE_FILE):
             print(f"\n✅ ĐÃ TẠO THÀNH CÔNG: {CACHE_FILE}")
             print("👉 BẠN ĐÃ CÓ THỨ CẦN THIẾT. HÃY DỪNG LẠI!")
             print(f"👉 Copy file '{CACHE_FILE}' xuống Jetson Orin Nano và chạy lệnh build ở đó.")
        
    except Exception as e:
        print(f"\n❌ Có lỗi (có thể bỏ qua nếu file cache đã được tạo): {e}")


if __name__ == "__main__":
    build_engine()