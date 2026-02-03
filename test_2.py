import onnx
from onnx import version_converter

model_path = "../onnxsim_romnistereo32_v13_bs16_e194.onnx"
original_model = onnx.load(model_path)

# Hạ cấp xuống opset 11 hoặc 12
target_opset = 11 
converted_model = version_converter.convert_version(original_model, target_opset)

onnx.save(converted_model, "../model_opset11.onnx")
print("Đã chuyển đổi sang Opset 11 thành công!")