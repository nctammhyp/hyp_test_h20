import onnxruntime
import numpy as np

# Đường dẫn file onnx vừa tạo
onnx_path = r"F:\algo\mvs_v119\checkpoints\model_pruned.onnx"

try:
    # 1. Load model
    session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("✅ Load model thành công!")

    # 2. Tạo input giả (Đúng shape và type như lúc export)
    # Lưu ý: Grid phải là float32
    # Shape ví dụ dựa trên log của bạn: Grid shape expected: (80, 320, 96, 2) -> Batch size = 1
    dummy_img = np.random.randn(1, 1, 768, 800).astype(np.float32) # Hoặc 3 kênh nếu dùng RGB
    dummy_grid = np.random.randn(80, 320, 96, 2).astype(np.float32)

    # 3. Chạy Inference
    inputs = {
        "img0": dummy_img,
        "img1": dummy_img,
        "img2": dummy_img,
        "grid0": dummy_grid,
        "grid1": dummy_grid,
        "grid2": dummy_grid
    }
    
    outputs = session.run(None, inputs)
    print("✅ Chạy Inference thành công!")
    print("Output shape:", outputs[0].shape)

except Exception as e:
    print("❌ Lỗi:", e)