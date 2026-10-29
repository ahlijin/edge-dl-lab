"""Mel 特征提取的 numpy 实现 —— 训练 / 离线评估 / 设备推理的唯一口径来源。

与下面这套 torchaudio 组合严格等价（实测与 librosa htk/norm=None 偏差 < 3e-4 dB）：

    MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400,
                   hop_length=160, n_mels=64, power=2.0)   # htk, norm=None, reflect pad
    AmplitudeToDB(top_db=None)                            # 绝对 dB
    x = x - x.max()                                       # 逐样本减最大值（去响度依赖）
    x = clamp(x, min=-80)                                 # 减完再截下限（等价 top_db=80）

设备端（Jetson 的 host_kws_test.py）直接 import 这里的 logmel_feat，
训练侧 common/audio/feature_extractor.py 也读这里的常量 —— 参数只改这一份。

注意：任何参数变动都必须重跑 projects/kws/cache_features.py，
否则训练用的还是旧口径的缓存（checkpoints/kws/*_cache.pt）。
"""

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64
TOP_DB = 80.0
AMIN = 1e-10
FEATURE_FLOOR = -TOP_DB   # 特征下限(dB)，同时是短片段补帧的填充值

_PAD = N_FFT // 2


def hz_to_mel(f):
    """HTK mel 刻度（torchaudio mel_scale="htk"）"""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def melscale_fbanks(n_mels=N_MELS, n_fft=N_FFT, sample_rate=SAMPLE_RATE,
                    f_min=0.0, f_max=None, norm=None):
    """torchaudio.functional.melscale_fbanks 的 numpy 版（mel_scale="htk"）。

    norm=None 表示不做面积归一（torchaudio MelSpectrogram 的默认，不是 librosa 的 slaney）。
    """
    if f_max is None:
        f_max = sample_rate / 2
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(f_min, f_max, n_freqs)
    m_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    if norm == "slaney":
        fb *= (2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels]))[None, :]
    return fb.astype(np.float32)


_MEL_FB = melscale_fbanks()
_WIN = np.hanning(WIN_LENGTH + 1)[:-1].astype(np.float32)   # periodic hann，同 torch.hann_window


def num_frames(num_samples):
    """torchaudio 的帧数公式：1 + (L + 2*pad - n_fft) // hop"""
    return 1 + (num_samples + 2 * _PAD - N_FFT) // HOP_LENGTH


def power_mel(wav):
    """波形 -> 功率 mel 谱，(N_MELS, T)，用于与 torch 实现逐点比对。"""
    y = np.asarray(wav, dtype=np.float32).squeeze()
    if y.ndim != 1:
        raise ValueError(f"power_mel 只接受单声道一维波形，得到 {y.shape}")
    mode = "reflect" if y.shape[0] > _PAD else "constant"
    yp = np.pad(y, (_PAD, _PAD), mode=mode)
    n_frames = num_frames(y.shape[0])
    frames = np.lib.stride_tricks.as_strided(
        yp, shape=(n_frames, N_FFT),
        strides=(yp.strides[0] * HOP_LENGTH, yp.strides[0]),
    )
    spec = np.abs(np.fft.rfft(frames * _WIN[None, :], n=N_FFT, axis=1)) ** 2
    return (spec @ _MEL_FB).T.astype(np.float32)


def to_relative_db(mel):
    """功率 mel -> 相对 dB：(n_mels, T)，最大值恒为 0，下限 -TOP_DB。"""
    db = 10.0 * np.log10(np.maximum(mel, AMIN))
    db = db - db.max()
    return np.maximum(db, FEATURE_FLOOR)


def logmel_feat(wav, pad_to=None):
    """波形 -> (N_MELS, T) float32 特征。pad_to 给定时不足则用 FEATURE_FLOOR 补帧。"""
    feat = to_relative_db(power_mel(wav))
    if pad_to is not None and feat.shape[1] < pad_to:
        feat = np.pad(feat, ((0, 0), (0, pad_to - feat.shape[1])),
                      mode="constant", constant_values=FEATURE_FLOOR)
    return feat.astype(np.float32)


def frames_per_clip(num_samples=SAMPLE_RATE):
    """1 秒 16k 音频的帧数（=101），供模型输入对齐用。"""
    return num_frames(num_samples)
