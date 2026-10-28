import torch
import sys, os
sys.path.append(os.path.abspath("../..")) # 解决路径问题
from projects.kws.model import DSCNN

# 1. 初始化模型并加载权重
model = DSCNN(n_classes=2)
model.load_state_dict(torch.load("best_wakeword.pth", map_location="cpu"))
model.eval()

# 2. 构造一个假的输入 (Batch=1, Channel=1, Freq=64, Time=101)
dummy_input = torch.randn(1, 1, 64, 101)

# 3. 导出 ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "wakeword.onnx", 
    input_names=["input"], 
    output_names=["output"], 
    opset_version=12,
    external_data=False
)
print("ONNX 导出成功！")