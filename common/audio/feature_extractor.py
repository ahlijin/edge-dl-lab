import torch
import torchaudio

from common.audio.feature_numpy import (
    SAMPLE_RATE, N_FFT, WIN_LENGTH, HOP_LENGTH, N_MELS, TOP_DB,
)


class AudioFeatureExtractor:
    """训练侧特征提取（torchaudio），参数与 numpy 版共用 common/audio/feature_numpy.py。

    口径：
      MelSpectrogram(htk, norm=None, power=2, reflect pad) -> 绝对 dB
      -> 逐样本减最大值（去掉响度 / AGC / 增益依赖）
      -> 截下限 -80（等价 top_db=80）

    先减最大值再截下限，是为了让 batch>1 时每条样本的截断也发生在
    「自己的最大值 - 80」上（torchaudio 的 AmplitudeToDB(top_db=80) 是拿整个 batch
    的全局最大值截断的），这样训练特征与设备端 batch=1 的特征逐点一致。
    """

    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS,
                 win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        # top_db=None：这里不截断，减完最大值后再按 TOP_DB 截，保证逐样本口径
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=None)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (num_samples,) 或 (batch, num_samples)，float32，范围 [-1, 1]
        返回: (n_mels, T)（单条输入）或 (batch, n_mels, T)（batch 输入）
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)    # (1, num_samples)
        mel = self.mel(waveform)                # (batch, n_mels, T)
        log_mel = self.to_db(mel)               # (batch, n_mels, T) 绝对 dB
        max_val = torch.amax(log_mel, dim=(-2, -1), keepdim=True)
        log_mel = log_mel - max_val             # 最大值恒为 0，范围 (-inf, 0]
        log_mel = torch.clamp(log_mel, min=-TOP_DB)
        return log_mel.squeeze(0)               # (n_mels, T)
