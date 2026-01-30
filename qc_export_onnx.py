import torch
import torchvision.models as models

# 1. Load model mặc định (ví dụ ResNet50)
model = models.resnet50(pretrained=True)
model.eval()

# 2. Tạo input giả lập (phải khớp với shape thực tế)
dummy_input = torch.randn(1, 3, 224, 224)

# Chỉnh lại đoạn này trong script export của bạn
torch.onnx.export(model, 
                  dummy_input, 
                  "model.onnx", 
                  export_params=True, 
                  opset_version=15,  # Chỉnh từ 17+ xuống 15
                  do_constant_folding=True)


import onnx

model_path = "model.onnx"
model = onnx.load(model_path)

# Hạ cấp IR version
if model.ir_version > 9:
    print(f"Đang hạ IR version từ {model.ir_version} xuống 9...")
    model.ir_version = 9
    onnx.save(model, model_path)
    print("Đã lưu lại model với IR version 9.")
else:
    print(f"Model đã ở IR version {model.ir_version}, không cần hạ cấp.")


