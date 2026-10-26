import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
import matplotlib.pyplot as plt
from common.audio.feature_extractor import AudioFeatureExtractor
from projects.kws.model import DSCNN

# 加载测试音频
wave_path = "D://test.wav"
# 直接用 soundfile 读取音频
# data 的形状是 (frames, channels)，dtype 为 float32
data, sr = sf.read(wave_path, dtype="float32", always_2d=True)

# 转置为 PyTorch 期望的 (channels, frames) 形状
waveform = torch.from_numpy(data.T).contiguous()

print(f"原始音频： shape={waveform.shape}, sr={sr}")

# 重采样到16kHz， 混单声道、截断/填充到1s
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
waveform = waveform.squeeze(0)
if waveform.shape[0] < 16000:
    waveform = F.pad(waveform, (0, 16000 - waveform.shape[0]))
else:
    waveform = waveform[:16000]
print(f"归一化后音频： shape={waveform.shape}")

# 特征提取
extractor = AudioFeatureExtractor()
feat = extractor(waveform)
print(f"特征图： shape={feat.shape}")  # (1, 64, 101)

# 模型前向
model = DSCNN(n_classes=12)
model.eval()
with torch.no_grad():
    logits = model(feat.unsqueeze(0).unsqueeze(1))
print(f"模型输出：shape={logits.shape}") # (1, 12)

# 参数量统计
n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量：{n_params:,}")