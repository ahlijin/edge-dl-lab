# edge-dl-lab

边缘设备深度学习练手项目集，目标是把每个项目完整走通：
数据 → 训练 → 量化 → 边缘部署 → 接入 ROS 2 / Zenoh 系统。

## 项目列表
- [x] 01 KWS：语音指令词分类（✅ 模型训练完成，指标达标，待量化与 ROS2 部署）
- [x] 02 Detection：轻量目标检测（✅ 训练/推理链路跑通，权重已入库，待评估指标、量化与部署）
- [ ] 03 E2E Nav：端到端视觉导航
- [ ] 04 Anomaly：时序异常检测

## 目录说明
- `common/`：跨项目复用的基础模块（目前只有 `audio/`：Mel 特征口径、numpy 实现、一致性自检）
- `projects/NN_xxx/`：每个练手项目的独立代码目录，内部统一分层：
  - `src/`：被 import 的模块（模型、数据集、损失、锚框等）
  - `scripts/`：可执行入口（数据准备 / 训练 / 分步验证 / 推理）
  - `deploy/`：设备端工具（kws 有 TensorRT 引擎封装与唤醒回归脚本）
  - `weights/`：入库的模型产物（`detector_weights.pth`、`best_wakeword.pth`、`wakeword.onnx`）
- `checkpoints/NN_xxx/`：机器本地的训练中间产物（目前只有 kws 的 `.pt` 特征缓存，已 gitignore、可重建）
- `notebooks/`、`docs/`：探索性实验与流程记录（按需创建）

## 快速开始
- 01 KWS：`python projects/kws/scripts/cache_features.py` → `python projects/kws/scripts/train_wakeword.py`，细节见 `projects/kws/README.md`
- 02 Detection：`python projects/detection/scripts/verify_step1.py` 起四步依次验证到 `predict.py`，细节见 `projects/detection/README.md`

## 版本与数据管理规范
- **代码入库**：各项目的 Python 脚本均放在 `projects/NN_xxx/` 下（可复用模块进 `src/`，入口脚本进 `scripts/`）。
- **模型入库**：小于 10MB 的轻量级模型产物（如 `projects/kws/weights/best_wakeword.pth`、`projects/detection/weights/detector_weights.pth`）直接提交至 Git；大于 10MB 的权重请使用 Git LFS 或 Gitea Release。
- **数据隔离**：原始数据集（音频/图像）与预处理缓存文件（`*.pt`）严禁提交，已在 `.gitignore` 中过滤，由各自项目的脚本自动生成。