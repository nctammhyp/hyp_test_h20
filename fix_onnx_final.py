import onnx

# Load model của bạn
model_path = r'F:\algo\mvs_v119\romnistereo_simplified.onnx'
model = onnx.load(model_path)

# Duyệt qua tất cả các node để tìm Reshape
for node in model.graph.node:
    if node.op_type == 'Reshape':
        # Tìm thuộc tính allowzero và sửa thành 0
        for attr in node.attribute:
            if attr.name == 'allowzero':
                attr.i = 0

# Lưu lại model đã sửa
onnx.save(model, r'F:\algo\mvs_v119\romni_fixed_reshape.onnx')
print("✅ Đã sửa thuộc tính allowzero về 0!")