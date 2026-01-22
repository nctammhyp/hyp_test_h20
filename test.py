import onnx

# Đường dẫn file ONNX của bạn
model_path = "./aimet_export/romni_quantized.onnx"
model = onnx.load(model_path)

print("--- INPUT NAMES IN ONNX ---")
for input in model.graph.input:
    print(f"Name: '{input.name}', Shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")