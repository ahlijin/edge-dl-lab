"""把训练好的 best_wakeword.pth 导出成 wakeword.onnx（给 Jetson 编 TensorRT engine）。

路径全部基于仓库根，从哪个目录跑都行：

    python checkpoints/kws/export_onnx.py
    python checkpoints/kws/export_onnx.py --pth checkpoints/kws/best_wakeword.pth --out checkpoints/kws/wakeword.onnx

注意：导出前模型必须处于 eval 模式（下面 model.eval() 之后有断言），
否则 BatchNorm 会用单个样本的统计量，模型导出后基本“不看内容”。

opset 默认 18：新版 torch 的 dynamo 导出器用 12 会报错；Jetson 侧 TensorRT 10.16
解析 opset 18 已验证可用（旧 engine 就是从 opset 18 的 onnx 编出来的）。

导出的 onnx 字节数只由计算图结构决定、与权重数值无关，所以重训前后大小一致是正常的，
不能用文件大小判断是不是新模型 —— 用 mtime 或实测概率判断。
"""

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints", "kws")

import torch

from projects.kws.src.model import DSCNN


def main():
    ap = argparse.ArgumentParser(description="导出 KWS 唤醒词模型为 ONNX")
    ap.add_argument("--pth", default=os.path.join(CKPT_DIR, "best_wakeword.pth"))
    ap.add_argument("--out", default=os.path.join(CKPT_DIR, "wakeword.onnx"))
    ap.add_argument("--opset", type=int, default=18,
                    help="ONNX opset，默认 18：新版 torch 的 dynamo 导出器对 12 会报错，"
                         "且 Jetson 上 TensorRT 10.16 已验证能直接编 opset 18")
    args = ap.parse_args()

    if not os.path.exists(args.pth):
        raise SystemExit(f"找不到权重 {args.pth}，先跑 projects/kws/scripts/train_wakeword.py")

    model = DSCNN(n_classes=2)
    model.load_state_dict(torch.load(args.pth, map_location="cpu"))
    model.eval()
    assert not model.training, "模型还在 train 模式，BatchNorm 会用 batch 统计量，禁止导出"

    # 输入固定为 (Batch=1, Channel=1, Freq=64, Time=101)：1 秒 16k 音频的 mel 特征
    dummy_input = torch.randn(1, 1, 64, 101)
    with torch.no_grad():
        out = model(dummy_input)
    assert tuple(out.shape) == (1, 2), f"输出形状异常: {tuple(out.shape)}"

    torch.onnx.export(
        model,
        dummy_input,
        args.out,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        external_data=False,
    )
    print(f"ONNX 导出成功: {args.out} ({os.path.getsize(args.out)} 字节)")
    print("下一步: 拷到 Jetson 后用 trtexec 重编 wakeword.engine（fp16），再跑 host_kws_test.py 回归")


if __name__ == "__main__":
    main()
