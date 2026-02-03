import onnx

# Đường dẫn file ONNX của bạn
model_path = r"F:\algo\mvs_v119\checkpoints\onnx\romnistereo32_v13_bs16_e194.onnx"
model = onnx.load(model_path)

print("--- INPUT NAMES IN ONNX ---")
for input in model.graph.input:
    print(f"Name: '{input.name}', Shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")


print("--- OUTPUT NAMES IN ONNX ---")
for output in model.graph.output:
    print(f"Name: '{output.name}', Shape: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
