import onnx
import onnx_graphsurgeon as gs
import numpy as np

model = onnx.load("romnistereo_v11_fixed.onnx")
graph = gs.import_onnx(model)

for node in graph.nodes:
    if node.op == "Unsqueeze":
        for i, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Constant):
                if inp.values.dtype == np.int64:
                    print("Fix Unsqueeze axis INT64 → INT32")
                    inp.values = inp.values.astype(np.int32)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model_fixed.onnx")
