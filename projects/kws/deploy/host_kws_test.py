"""Jetson 端 KWS 实时测试：ReSpeaker 采集 -> 特征 -> TensorRT 推理 -> 打概率。

特征口径不在这里实现，直接 import 仓库里的 projects/kws/src/audio/feature_numpy.py（部署时放在同目录）。
训练侧（torchaudio）读的是同一份参数，所以这里不需要再对一遍。

部署（两行，都在仓库根执行）：
    scp projects/kws/src/audio/feature_numpy.py nvidia@<jetson>:~/hijetson/
    scp projects/kws/deploy/host_kws_test.py    nvidia@<jetson>:~/hijetson/

运行（在 Jetson 的 ~/hijetson 下，需要 wakeword.engine）：
    .venv/bin/python host_kws_test.py
"""

import os
import sys
import time
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

# 特征实现与引擎封装都与仓库共用一份（部署时和本文件放同目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_numpy import (  # noqa: E402
    SAMPLE_RATE, N_MELS, TOP_DB, FEATURE_FLOOR, logmel_feat, frames_per_clip,
)
from trt_engine import TRTEngine  # noqa: E402

# ================= 配置参数 =================
ENGINE_PATH = "wakeword.engine"
BLOCK_SEC = 0.5
THRESHOLD = 0.6         # 重训后必须在本机实录音上重新定，别沿用旧模型调出来的值
CONSECUTIVE_HITS = 2    # 1s 窗 / 0.5s 步进，一个词只有 2~3 帧
SAVE_RMS = 0.010        # 存文件门限(RMS)；ReSpeaker 静音 rms≈0.0015，说话 0.02~0.1
SAVE_COOLDOWN = 3.0     # 两次存文件最小间隔(秒)
DEVICE_NAME_HINT = "ReSpeaker"

N_FRAMES = frames_per_clip(SAMPLE_RATE)   # 1 秒 -> 101 帧
# 特征已做逐样本电平归一（最大值恒为 0），所以不再需要 GAIN 之类的增益补偿：
# 任何常数增益都会被减最大值整体抵消。AGC 开/关同理，不再是生死线。


def find_respeaker():
    """按名字找 ReSpeaker 索引，绝不依赖 ALSA 默认设备
    （/etc/asound.conf 把 default 指到 hw:APE,0，那里没接麦克风，采到全 0）。"""
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and DEVICE_NAME_HINT.lower() in d["name"].lower():
            return i
    return None


# ================= 主程序 =================
print("正在加载 TensorRT 引擎...")
engine = TRTEngine(ENGINE_PATH)
print("引擎加载成功！")

DEVICE_IDX = find_respeaker()
if DEVICE_IDX is None:
    raise SystemExit("找不到 ReSpeaker 麦克风，检查 lsusb | grep 2886 / arecord -l")
print(f"麦克风设备: [{DEVICE_IDX}] {sd.query_devices(DEVICE_IDX)['name']}")
print(f"特征口径: n_mels={N_MELS} 帧数={N_FRAMES} 值域=[{FEATURE_FLOOR:g}, 0] "
      f"(逐样本减最大值, top_db={TOP_DB:g})")

buffer = np.zeros(SAMPLE_RATE, dtype=np.float32)
hit_count = 0
_last_save = 0.0


def _save_async(buf, rms):
    """存盘放后台线程，别拖慢音频回调。"""
    def _w():
        fn = time.strftime("recorded_%H%M%S.wav")
        sf.write(fn, buf.copy(), SAMPLE_RATE)
        print(f"已保存录音: {fn} (rms={rms:.4f})")
    threading.Thread(target=_w, daemon=True).start()


def audio_callback(indata, frames, time_info, status):
    global buffer, hit_count, _last_save
    if status:
        print(f"音频错误: {status}")

    new_audio = indata[:, 0]
    buffer = np.roll(buffer, -len(new_audio))
    buffer[-len(new_audio):] = new_audio

    rms = float(np.sqrt(np.mean(buffer ** 2)))
    peak = float(np.max(np.abs(buffer)))

    # 存文件：RMS 门限 + 冷却时间（peak>0.5 那种条件静音时永远触发不了）
    now = time.time()
    if rms > SAVE_RMS and (now - _last_save) > SAVE_COOLDOWN:
        _last_save = now
        _save_async(buffer, rms)

    # 特征提取（与训练同一份实现）
    feat2d = logmel_feat(buffer, pad_to=N_FRAMES)
    feat = feat2d[np.newaxis, np.newaxis, :, :]

    # 推理
    t0 = time.time()
    logits = engine.infer(feat)
    latency = (time.time() - t0) * 1000

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    wake_prob = float(probs[1])

    if wake_prob > THRESHOLD:
        hit_count += 1
    else:
        if hit_count > 0:
            print(f"❌ 中断，概率不足: {wake_prob:.3f}")
        hit_count = 0

    print(f"当前概率: {wake_prob:.3f} | rms: {rms:.4f} peak: {peak:.4f} "
          f"| featmax: {feat2d.max():.1f} featmin: {feat2d.min():.1f} | 延迟: {latency:.1f}ms")

    if hit_count >= CONSECUTIVE_HITS:
        print("🔔 唤醒成功！")
        hit_count = 0


block_size = int(SAMPLE_RATE * BLOCK_SEC)
print(f"开始监听麦克风，请说出唤醒词... (THRESHOLD={THRESHOLD}, HITS={CONSECUTIVE_HITS})")
try:
    with sd.InputStream(device=DEVICE_IDX, channels=1, samplerate=SAMPLE_RATE,
                        blocksize=block_size, callback=audio_callback):
        while True:
            time.sleep(1)
except KeyboardInterrupt:
    print("\n停止监听。")