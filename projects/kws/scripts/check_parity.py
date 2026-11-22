"""特征口径自检：numpy 实现 vs torchaudio 实现，逐点比对。

训练前（或改过 feature_numpy.py 之后）跑一遍，确认设备端会得到和训练一样的特征：

    python projects/kws/scripts/check_parity.py <wav1> [<wav2> ...]
    python projects/kws/scripts/check_parity.py --random 3        # 用随机噪声自测

判定标准：功率谱相对误差 < 1e-3，最终特征最大绝对偏差 < 1e-2 dB。
不通过就别往下跑 cache_features.py。
"""

import os
import sys

# 将项目根目录加入 sys.path，解决 import 报错
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, ROOT_DIR)

import argparse

import numpy as np

from projects.kws.src.audio import feature_numpy as fn


def _torch_parts(wav_np):
    import torch
    import torchaudio

    from projects.kws.src.audio.feature_extractor import AudioFeatureExtractor

    wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32))
    ext = AudioFeatureExtractor()
    with torch.no_grad():
        mel = ext.mel(wav.unsqueeze(0)).squeeze(0)          # (n_mels, T) 功率
        feat = ext(wav)                                     # (n_mels, T) 相对 dB
    return mel.numpy(), feat.numpy()


def check_one(name, wav):
    mel_np = fn.power_mel(wav)
    feat_np = fn.logmel_feat(wav)
    mel_t, feat_t = _torch_parts(wav)

    if mel_np.shape != mel_t.shape:
        print(f"[FAIL] {name}: 形状不一致 numpy{mel_np.shape} vs torch{mel_t.shape}")
        return False

    rel = np.abs(mel_np - mel_t) / np.maximum(mel_t, 1e-10)
    feat_diff = np.abs(feat_np - feat_t)
    ok = rel.max() < 1e-3 and feat_diff.max() < 1e-2
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: 形状{mel_np.shape} "
          f"功率谱相对误差max={rel.max():.2e} 特征偏差max={feat_diff.max():.4f}dB "
          f"值域[{feat_np.min():.1f},{feat_np.max():.1f}]")
    return ok


def main():
    ap = argparse.ArgumentParser(description="numpy 特征实现与 torchaudio 实现的逐点比对")
    ap.add_argument("wavs", nargs="*", help="待比对的 wav 文件（16k 单声道最佳）")
    ap.add_argument("--random", type=int, default=0, help="额外用 N 段随机/正弦自测")
    args = ap.parse_args()

    if not args.wavs and not args.random:
        ap.error("至少给一个 wav，或用 --random 3 自测")

    import soundfile as sf

    ok_all = True
    for path in args.wavs:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != fn.SAMPLE_RATE:
            print(f"[SKIP] {path}: 采样率 {sr} != {fn.SAMPLE_RATE}，先重采样再比")
            ok_all = False
            continue
        ok_all &= check_one(path, wav)

    rng = np.random.default_rng(0)
    for i in range(args.random):
        if i % 2 == 0:
            wav = rng.standard_normal(fn.SAMPLE_RATE).astype(np.float32) * 0.05
            tag = f"random-{i}"
        else:
            t = np.arange(fn.SAMPLE_RATE) / fn.SAMPLE_RATE
            wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            tag = f"sine-{i}"
        ok_all &= check_one(tag, wav)

    print("\n结果:", "通过，设备端与训练口径一致" if ok_all else "不通过，先修特征实现再缓存/训练")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
