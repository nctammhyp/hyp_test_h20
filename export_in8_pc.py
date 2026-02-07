import tensorrt as trt
import os

# ================= CẤU HÌNH PC =================
ONNX_PATH = "checkpoints/onnx/romnistereo32_v20_bs8_e46_jetson.onnx"
# Đổi tên để phân biệt với bản Jetson
ENGINE_PATH_PC = "checkpoints/onnx/romnistereo32_v20_bs8_e46_3060_int8.engine"
CACHE_FILE = "calib.cache" 
BATCH_SIZE = 1 # Bạn có thể tăng lên nếu muốn batch lớn hơn trên PC

# PC có 12GB VRAM nên có thể để Workspace thoải mái hơn
WORKSPACE_SIZE = 2048 * 1024 * 1024  # 2GB

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Calibrator tối giản: Chỉ đọc cache, không cần data gốc
class CacheCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file):
        super().__init__()
        self.cache_file = cache_file

    def get_batch_size(self): return BATCH_SIZE
    def get_batch(self, names): return None # Không cần data vì đã có cache
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print(f"--> Đang sử dụng cache: {self.cache_file}")
                return f.read()
        return None
    def write_calibration_cache(self, cache): pass

def build_engine_pc():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 1. Parse ONNX
    print("-> Parsing ONNX...")
    with open(ONNX_PATH, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 2. Config tối ưu cho RTX 3060
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE)
    
    # Kích hoạt cả INT8 và FP16
    if builder.platform_has_fast_int8:
        print("✅ PC hỗ trợ INT8. Đang thiết lập...")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = CacheCalibrator(CACHE_FILE)
    
    if builder.platform_has_fast_fp16:
        print("✅ Kích hoạt thêm FP16 Fallback.")
        config.set_flag(trt.BuilderFlag.FP16)

    # 3. Build
    print(f"-> Đang build Engine cho RTX 3060: {ENGINE_PATH_PC}")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine:
            with open(ENGINE_PATH_PC, "wb") as f:
                f.write(serialized_engine)
            print("--- THÀNH CÔNG ---")
        else:
            print("--- THẤT BẠI (Engine rỗng) ---")
    except Exception as e:
        print(f"Lỗi khi build: {e}")

if __name__ == "__main__":
    build_engine_pc()