# 01-KWS：中文唤醒词检测 (Keyword Spotting)

基于 MobvoiHotwords 数据集训练的中文唤醒词二分类模型。主要目标是验证从数据处理、模型训练到边缘设备部署的全链路流程。

## 模型信息
- **网络架构**：DSCNN (Depthwise Separable CNN)
- **参数量**：52,716 (约 52K)
- **模型大小**：约 200 KB (`.pth`)
- **输入特征**：`1 x 64 x 101` (1 秒音频提取的 Mel 频谱图)
- **输出类别**：2 (0 = 非唤醒词 / 负样本, 1 = 唤醒词 / 正样本)

## 输入特征口径（改口径必须重跑缓存）

训练、离线评估、设备推理必须是同一套口径，唯一来源是 `projects/kws/src/audio/feature_numpy.py`：

1. `MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160,
   n_mels=64, power=2.0)` —— **htk** mel、**不做面积归一**(`norm=None`)、reflect 填充、periodic hann 窗
2. 转 dB：`10 * log10(max(mel, 1e-10))`（绝对 dB）
3. **逐样本减去本样本的最大值** —— 让最大值恒为 0，彻底去掉对响度 / 麦克风增益 / AGC / 说话距离的依赖
4. 减完再截下限 `-80 dB`（等价 `top_db=80`）

所以特征值域是 `[-80, 0]`，**不含绝对响度信息**。
推论：设备端的 `GAIN` 常数、AGC 开/关都不再影响特征（任何常数增益都被第 3 步整体抵消）；
短片段补帧要填 `-80`（填 0 等于给时间轴补"最强能量"帧）。

⚠️ 只要上面任一条变了，`checkpoints/kws/` 下三份缓存 `train/val/test_cache.pt` 全部作废，
必须重跑 `cache_features.py`（`train_wakeword.py` 开头有缓存新鲜度检查，会拦住"改了特征忘了重建缓存"）。

## 数据集与依赖
- 数据集：[MobvoiHotwords](https://openslr.org/87/) (SLR87)
- 依赖库：`torch`, `torchaudio`, `soundfile`, `tqdm`, `numpy`
*(注：音频读取使用 `soundfile`，彻底绕过 Windows 下 `torchaudio` 依赖 `ffmpeg` 的报错)*

## 目录结构约定
```text
edge-dl-lab/
├── checkpoints/kws/           # 自动生成，只放 .pt 特征缓存（3.7GB，可重建，已 gitignore）
└── projects/kws/              # 训练相关代码 + 入库产物
    ├── src/
    │   ├── model.py                   # DSCNN
    │   └── audio/
    │       ├── feature_numpy.py       # 特征口径唯一来源（numpy，设备端直接 import 这份）
    │       └── feature_extractor.py   # torchaudio 版，参数从 feature_numpy.py 取
    ├── scripts/               # 可执行入口：cache_features.py / train_wakeword.py / export_onnx.py / check_parity.py
    ├── weights/               # 入库的模型产物：best_wakeword.pth、wakeword.onnx
    └── deploy/                # 设备端工具：trt_engine.py（引擎封装）+ host_kws_test.py（唤醒回归）
```

## 全流程

```bash
# 0) 特征口径自检（numpy 设备实现 vs torchaudio 训练实现，偏差应 < 1e-2 dB）
python projects/kws/scripts/check_parity.py ~/clips/唤醒词1.wav --random 2

# 1) 重建三份特征缓存（数据目录用参数或环境变量，别再写死在代码里）
python projects/kws/scripts/cache_features.py \
    --json-dir  <标注目录，含 p_train.json / n_train.json ...> \
    --audio-dir <wav 目录>
#   或：export MOBVOI_JSON_DIR=... MOBVOI_AUDIO_DIR=...

# 2) 训练（读缓存；缓存比特征代码旧会直接报错退出）
python projects/kws/scripts/train_wakeword.py

# 3) 导出 ONNX（eval 模式，输入 1x1x64x101，输出到 projects/kws/weights/wakeword.onnx）
python projects/kws/scripts/export_onnx.py

# 4) 拷到 Jetson，编 TensorRT engine 后跑 host_kws_test.py 回归（已在本机 Jetson 上验证通过）
#    trtexec --onnx=wakeword.onnx --saveEngine=wakeword.engine --fp16

# 5) 设备端唤醒回归工具 host_kws_test.py（只打唤醒概率，不做命令识别）
#    部署：scp projects/kws/src/audio/feature_numpy.py projects/kws/deploy/{trt_engine,host_kws_test}.py nvidia@<jetson>:~/hijetson/
#    运行：.venv/bin/python host_kws_test.py
```

## 部署状态
- **已通过（实机 Jetson，精度 fp16）**：`best_wakeword.pth` → `wakeword.onnx`（opset 18）→ `trtexec --fp16` 编 TensorRT engine → `host_kws_test.py` 回归 → ROS 2 节点接入，整条链路唤醒概率输出正常。
- engine 与 onnx 的输入形状都固定 `1x1x64x101`，量化只用 fp16（未做 INT8 校准）。
- ROS 2 侧的节点 / launch 代码在设备端工作区，不在本仓库；本仓库提供的是特征实现（`projects/kws/src/audio/feature_numpy.py`）与引擎封装 + 回归脚本（`projects/kws/deploy/`）。

## 评估要求
- **两套都要看**：① Mobvoi test set（确认没忘本，脚本已打印 0.5/0.6/0.7/0.9 各阈值召回率与 FAR）；
  ② 本机 ReSpeaker 实录（房间环境、20cm/1m/3~5m 各距离），这才是验收，按距离/电平分桶统计召回。
- 阈值 `THRESHOLD` / `CONSECUTIVE_HITS` 必须在本机实录音上重新定，别沿用旧模型调出来的值。

## 已知坑
- 特征口径（mel 刻度、归一方式）与训练不一致时，模型对真实人声会给恒 0 概率 —— 先跑 `check_parity`。
- `best_wakeword.pth` 若不跟着特征改动重训，设备端"新特征 + 旧权重"会完全失效（值域整体平移几十 dB）。
- 导出 ONNX 时模型必须是 `eval()` 模式（`export_onnx.py` 已断言）。