import torch
import torchaudio

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=64, win_length=400, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mels= n_mels
        self.mel= torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db=torchaudio.transforms.AmplitudeToDB(top_db=80)
        # 归一化统计量，训练后填入，这里先用占位值
        self.mean = 0.0
        self.std = 1.0

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """"
        waveform: (num_samples,) 或 (batch, num_samples)，float32，范围 [-1, 1]
        返回: (1, n_mels, T) 的 Log-Mel 特征图
        """
        if waveform.dim()==1:
            waveform = waveform.unsqueeze(0)    # (1, num_samples)
        mel = self.mel(waveform)                # (batch, n_mels, T)
        log_mel = self.to_db(mel)               # (batch, n_mels, T)
        log_mel = (log_mel - self.mean) / (self.std + 1e-6)
        return log_mel.squeeze(0)               # (n_mels, T)