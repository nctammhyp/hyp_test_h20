import onnx

# Input: File ONNX vừa tạo từ AIMET (chưa qua converter)
INPUT_MODEL = r"checkpoints/model_mvs_h20_clean.onnx"
OUTPUT_MODEL = r"romni_patched.onnx"

print(f"Loading {INPUT_MODEL}...")
try:
    model = onnx.load(INPUT_MODEL)
except Exception as e:
    print(f"❌ Cannot load model: {e}")
    exit(1)

print("Patching Reshape 'allowzero' attribute...")
count = 0
for node in model.graph.node:
    if node.op_type == "Reshape":
        for attr in node.attribute:
            if attr.name == "allowzero":
                if attr.i == 1:
                    attr.i = 0
                    count += 1

print(f"✅ Patched {count} Reshape nodes.")

# Lưu ý: Cập nhật luôn Opset Version về 16 hoặc 17 (QAIRT hỗ trợ tốt hơn 18)
# nếu model cho phép.
if model.opset_import[0].version > 17:
    print(f"Downgrading Opset {model.opset_import[0].version} -> 17")
    model.opset_import[0].version = 17

onnx.save(model, OUTPUT_MODEL)
print(f"Saved CLEAN model to: {OUTPUT_MODEL}")