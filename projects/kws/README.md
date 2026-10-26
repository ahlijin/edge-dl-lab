# 01-KWS：中文唤醒词检测 (Keyword Spotting)

基于 MobvoiHotwords 数据集训练的中文唤醒词二分类模型。主要目标是验证从数据处理、模型训练到边缘设备部署的全链路流程。

## 模型信息
- **网络架构**：DSCNN (Depthwise Separable CNN)
- **参数量**：52,716 (约 52K)
- **模型大小**：约 200 KB (`.pth`)
- **输入特征**：`1 x 64 x 101` (1 秒音频提取的 Mel 频谱图)
- **输出类别**：2 (0 = 非唤醒词 / 负样本, 1 = 唤醒词 / 正样本)

## 数据集与依赖
- 数据集：[MobvoiHotwords](https://openslr.org/87/) (SLR87)
- 依赖库：`torch`, `soundfile`, `tqdm`, `numpy`
*(注：音频读取使用 `soundfile`，彻底绕过 Windows 下 `torchaudio` 依赖 `ffmpeg` 的报错)*

## 目录结构约定
在使用脚本前，确保目录结构如下：
```text
edge-dl-lab/
├── mobvoi_json/               # 数据集 json 标注文件目录
├── mobvoi_hotword_dataset/    # 数据集 wav 音频解压目录
├── checkpoints/kws/           # 自动生成，用于存放 .pt 缓存和 .pth 权重
└── projects/kws/              # 本目录，存放训练代码