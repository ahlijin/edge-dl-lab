# 02-Detection：轻量目标检测

基于 HomeObjects-3K（YOLO 格式标注）手搓的单尺度 Anchor-based 检测器。
目标是把检测的整条链路拆成可以**逐步验证**的小步：锚框 → 标签分配 → 模型 + Loss → 训练 → 推理，
每一步都能单独跑一遍看图，而不是一次性训练完再看结果。

## 模型信息
- **网络架构**：TinyDetector（Depthwise Separable 主干 + 三个 1×1 检测头）
- **输入**：`3 x 320 x 320`，主干 stride=8，输出特征图 `40 x 40`
- **检测头**：回归 `4*A`、分类 `A*12`、置信度 `A*1`（A = 每个位置的锚框数）
- **权重**：`weights/detector_weights.pth`（约 1 MB，直接入库）

## 锚框与标签分配
- **锚框**：`generate_anchors((40, 40), scales=[32, 64, 128, 256], ratios=[0.5, 1.0, 2.0], stride=8)`
  → 每个位置 12 个锚框，共 `40*40*12 = 19200` 个
  （`verify_step1~3` 用的是 `scales=[32, 64]`，6 个/位置 = 9600 个，用来快速看图）
- **分配规则**（`AnchorAssigner(pos_iou_thresh=0.5, neg_iou_thresh=0.3)`）：
  - 与某个 GT 的 `IoU >= 0.5` → 正样本（objectness = 1）
  - `0.3 <= IoU < 0.5` → 忽略（objectness = -1，不计 loss）
  - 其余 → 负样本（objectness = 0）
  - **兜底**：每个 GT 至少保一个正样本，但只给"还没有正样本"的 GT 兜底，
    并且只在**尚未被占用**的锚框里挑 IoU 最高的。
    （否则两个 GT 抢同一个锚框时，后来的会把先来的覆盖掉，被抢的那个 GT 一个正样本都不剩）
- **编码**：`tx=(gx-ax)/aw`、`ty=(gy-ay)/ah`、`tw=log(gw/aw)`、`th=log(gh/ah)`，解码用同样的逆变换

## 损失
`DetectionLoss` = `BCEWithLogits(objectness)` + `CrossEntropy(分类)` + `SmoothL1(回归)`，权重 1:1:1；
置信度项只对非忽略样本求均值，分类与回归只对正样本求均值。

## 目录结构
```text
projects/detection/
├── src/                     # 被 import 的模块
│   ├── anchor_utils.py      # 锚框生成 + IoU
│   ├── dataset.py           # HomeObjectsDataset（读 YOLO 标签 → (cx,cy,w,h) 绝对坐标）
│   ├── target_assigner.py   # AnchorAssigner（正/负/忽略 + 每 GT 兜底）
│   ├── model.py             # TinyDetector
│   └── loss.py              # DetectionLoss
├── scripts/                 # 可执行入口
│   ├── verify_step1.py      # 锚框 vs GT 可视化
│   ├── verify_step2.py      # 标签分配结果（正/负/忽略 + 编码解码还原）
│   ├── verify_step3.py      # 模型输出形状 + 初始 loss
│   ├── train.py             # 训练，权重存到 weights/
│   └── predict.py           # 加载权重推理 + NMS + 可视化
├── weights/                 # 入库的权重（.pth）
└── data/homeobjects-3K/     # 数据集（images/、labels/，已被 .gitignore 过滤）
```

脚本里的 `current_dir` 都是 `projects/detection`（由 `__file__` 上溯两层得到），
所以 `data/`、`weights/` 都相对它定位，从哪个目录执行都可以。
IDE 的静态解析靠仓库根目录 `pyrightconfig.json` 里的 `extraPaths`（`projects/detection/src`）。

## 全流程

```bash
# 1) 锚框 + GT 可视化（先确认锚框尺度和物体大小匹配）
python projects/detection/scripts/verify_step1.py

# 2) 标签分配可视化（正样本解码回来应该和红框重合）
python projects/detection/scripts/verify_step2.py

# 3) 模型前向 + 初始 loss（随机权重，看形状和量级是否正常）
python projects/detection/scripts/verify_step3.py

# 4) 训练（AdamW lr=1e-3、Cosine 25 epoch、batch=4、梯度裁剪 10）
python projects/detection/scripts/train.py

# 5) 推理（score 阈值 0.3 + NMS IoU 0.45，红框 GT / 绿框预测）
python projects/detection/scripts/predict.py
```

## 已知坑
- `TinyDetector(num_anchors=...)` 必须和 `generate_anchors` 生成的数量一致：
  训练/推理是 12（4 尺度 × 3 比例），`verify_step1~3` 是 6（2 尺度 × 3 比例）。
  不一致会直接报形状错误 —— 脚本里都留着切换用的注释行。
- 锚框和 GT 必须在**同一个 device** 上，否则 GPU 训练时 IoU 计算报设备不一致。
- `data/` 被 `.gitignore` 过滤（原始图像 + 标签），换机器要重新放数据；
  几个验证脚本都是 `plt.show()` 直接弹窗，不落盘（无显示环境时设 `MPLBACKEND=Agg` 只能跑通、看不到图）。
- 改 `AnchorAssigner` 的兜底逻辑后，建议先用 `verify_step2.py` 过一遍
  （曾经出现过"两个 GT 抢同一个锚框 → 后面那个 GT 零正样本、整张图没梯度"的问题）。
