import onnx
import os

input_model = r"F:\algo\mvs_v119\checkpoints\model_mvs_h20.onnx"
output_model = r"F:\algo\mvs_v119\checkpoints\model_mvs_h20_clean.onnx"

print(f"Loading {input_model}...")
# 1. Load model (lúc này nó sẽ đọc file .data bên cạnh)
model = onnx.load(input_model)

print("Checking model...")
try:
    # Kiểm tra xem cấu trúc model có hợp lệ không
    onnx.checker.check_model(model)
    print("✅ Model structure is valid.")
except Exception as e:
    print(f"⚠️ Warning: Model check failed, but we will try to save anyway. Error: {e}")

print(f"Saving to {output_model}...")
# 2. Lưu lại model
# Hành động này sẽ buộc thư viện ONNX viết lại đường dẫn external data 
# thành đường dẫn tương đối (relative path) chuẩn Linux.
onnx.save(model, output_model)

print("Done! Use 'model_pruned_clean.onnx' for QAIRT.")