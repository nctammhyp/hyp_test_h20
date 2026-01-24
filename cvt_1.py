import onnx
import onnx.utils
from onnx import helper

INPUT_MODEL = "romni_final_for_htp.onnx"
OUTPUT_MODEL = "romni_final_fixed_sorted.onnx"

print(f"Loading {INPUT_MODEL}...")
model = onnx.load(INPUT_MODEL)

# 1. Sửa lỗi 'allowzero' cho Reshape (Lỗi kinh điển của QAIRT)
print("Step 1: Patching Reshape allowzero...")
for node in model.graph.node:
    if node.op_type == "Reshape":
        for attr in node.attribute:
            if attr.name == "allowzero":
                attr.i = 0

# 2. Sửa lỗi tên node 'unsqueeze' gây KeyError
# Đổi tên tất cả các node và output nếu chúng trùng với tên toán tử
print("Step 2: Renaming confusing nodes...")
for i, node in enumerate(model.graph.node):
    if node.name == "" or node.name == "unsqueeze" or node.name == "grid_sampler":
        node.name = f"node_{node.op_type}_{i}"
    
    # Đổi tên output nếu nó bị đặt là 'unsqueeze'
    for j, output_name in enumerate(node.output):
        if output_name == "unsqueeze":
            new_name = f"buffer_unsqueeze_{i}"
            print(f" -> Renaming output 'unsqueeze' to '{new_name}'")
            # Cập nhật mọi nơi sử dụng output này làm input
            for consumer in model.graph.node:
                for k, input_name in enumerate(consumer.input):
                    if input_name == "unsqueeze":
                        consumer.input[k] = new_name
            node.output[j] = new_name

# 3. Ép buộc sắp xếp lại thứ tự (Topological Sort)
# Đây là bước sửa lỗi 'is not output of any previous nodes'
print("Step 3: Performing Topological Sort...")
try:
    model = onnx.utils.topological_sort(model)
    print("✅ Topological sort successful.")
except Exception as e:
    print(f"❌ Sort failed: {e}")

# 4. Hạ cấp Opset xuống 11 (Nếu đang cao hơn)
# Giúp loại bỏ các thuộc tính không tương thích của Unsqueeze/Expand
print("Step 4: Ensuring Opset 11 compatibility...")
for opset in model.opset_import:
    if opset.domain == "" or opset.domain == "ai.onnx":
        opset.version = 11

print(f"Saving cleaned model to {OUTPUT_MODEL}...")
onnx.save(model, OUTPUT_MODEL)
print("Done! Now try converting this file.")