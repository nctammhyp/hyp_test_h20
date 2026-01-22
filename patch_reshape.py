import onnx

# Tên file ONNX hiện tại của bạn
INPUT_MODEL = r"F:\algo\mvs_v119\checkpoints\model_mvs_h20_clean.onnx"
# Tên file sau khi sửa
OUTPUT_MODEL = r"F:\algo\mvs_v119\checkpoints\model_mvs_h20_final.onnx"

print(f"Loading {INPUT_MODEL}...")
model = onnx.load(INPUT_MODEL)

print("Patching Reshape 'allowzero' attribute...")
count = 0
for node in model.graph.node:
    if node.op_type == "Reshape":
        for attr in node.attribute:
            if attr.name == "allowzero":
                # QAIRT chỉ hỗ trợ allowzero = 0
                if attr.i == 1:
                    print(f" -> Fixing node '{node.name}': allowzero 1 -> 0")
                    attr.i = 0
                    count += 1

if count == 0:
    print("No Reshape nodes with allowzero=1 found.")
else:
    print(f"✅ Successfully patched {count} Reshape nodes.")

print(f"Saving to {OUTPUT_MODEL}...")
onnx.save(model, OUTPUT_MODEL)
print("Done.")