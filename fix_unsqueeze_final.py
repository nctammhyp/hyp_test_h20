import onnx
from onnx import helper, numpy_helper
import numpy as np

INPUT = "./aimet_export/romni_quantized.onnx"
OUTPUT = "./aimet_export/romni_patched_final.onnx"

print(f"Loading {INPUT}...")
model = onnx.load(INPUT)
graph = model.graph

nodes_to_remove = []
nodes_to_add = []
initializers = {init.name: init for init in graph.initializer}

print("Patching Unsqueeze nodes...")
count = 0

for node in graph.node:
    if node.op_type == "Unsqueeze":
        # Kiểm tra xem input thứ 2 (axes) có phải là constant không
        if len(node.input) > 1 and node.input[1] in initializers:
            axes_tensor = initializers[node.input[1]]
            axes_val = numpy_helper.to_array(axes_tensor).tolist()
            if isinstance(axes_val, int): axes_val = [axes_val]
            
            # Tạo node Unsqueeze mới với axes là attribute (Opset 11 style)
            # HTP hỗ trợ kiểu này tốt hơn nhiều
            new_node = helper.make_node(
                "Unsqueeze",
                inputs=[node.input[0]], # Chỉ lấy input data, bỏ input axes
                outputs=node.output,
                name=node.name + "_patched",
                axes=axes_val 
            )
            
            nodes_to_remove.append(node)
            nodes_to_add.append(new_node)
            count += 1

# Thay thế node
for node in nodes_to_remove:
    graph.node.remove(node)
for node in nodes_to_add:
    graph.node.append(node)

# Cập nhật Opset Version về 11 (để hợp lệ với Unsqueeze static)
found_opset = False
for op in model.opset_import:
    if op.domain == "" or op.domain == "ai.onnx":
        op.version = 11
        found_opset = True

if not found_opset:
    model.opset_import.append(helper.make_opsetid("", 11))

print(f"✅ Patched {count} Unsqueeze nodes to Static Opset 11.")
onnx.save(model, OUTPUT)
print(f"Saved to {OUTPUT}")