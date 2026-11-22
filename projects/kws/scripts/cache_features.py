import os
import sys

# 将项目根目录加入 sys.path，解决 import 报错
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, ROOT_DIR)

import json
import random
import torch
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from projects.kws.src.audio.feature_extractor import AudioFeatureExtractor
from projects.kws.src.audio.feature_numpy import FEATURE_FLOOR, N_MELS, TOP_DB

class MobvoiDataset(Dataset):
    def __init__(self, json_files, audio_dir, extractor, target_len=16000):
        self.items = []
        self.audio_dir = audio_dir
        self.extractor = extractor
        self.target_len = target_len
        
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    label = 1 if item["keyword_id"] >= 0 else 0
                    self.items.append((item["utt_id"], label))
                    
    def __len__(self):
        return len(self.items)

    def load_audio(self, path, label):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T).mean(dim=0) # 混单声道
        
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, 16000).squeeze(0)
            
        if wav.shape[0] < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - wav.shape[0]))
        else:
            if label == 1:
                # 正样本用能量最大窗口
                stride = 800
                squared = wav ** 2
                cs = torch.cumsum(squared, dim=0)
                cs = torch.cat([torch.tensor([0.0]), cs])
                starts = torch.arange(0, wav.shape[0] - self.target_len + 1, stride)
                ends = starts + self.target_len
                energies = cs[ends] - cs[starts]
                best_start = starts[torch.argmax(energies)].item()
                wav = wav[best_start:best_start+self.target_len]
            else:
                # 负样本缓存时全部固定从中间截取，训练时靠打乱数据增强
                wav = wav[:self.target_len]
        return wav

    def __getitem__(self, idx):
        utt_id, label = self.items[idx]
        path = os.path.join(self.audio_dir, f"{utt_id}.wav")
        wav = self.load_audio(path, label)
        
        feat = self.extractor(wav)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        if feat.shape[-1] < 101:
            # 用特征下限(-80dB)补帧：现在 0 等于“最大能量”，补 0 会给短片段凭空加几帧最强能量
            feat = torch.nn.functional.pad(feat, (0, 101 - feat.shape[-1]), value=FEATURE_FLOOR)
        elif feat.shape[-1] > 101:
            feat = feat[:, :, :101]
            
        return feat, label

def extract_and_save(jsons, save_name, audio_dir, workers=8):
    print(f"正在处理并缓存 {save_name}...")
    ds = MobvoiDataset(jsons, audio_dir, AudioFeatureExtractor())
    # 这里用多线程加速读取 wav
    loader = DataLoader(ds, batch_size=256, num_workers=workers)
    
    all_feats = []
    all_labels = []
    
    for feat, label in tqdm(loader):
        all_feats.append(feat.half())  # 转成 float16 节省一半内存
        all_labels.append(label)
        
    feats_tensor = torch.cat(all_feats, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    torch.save({"feats": feats_tensor, "labels": labels_tensor}, save_name)
    print(f"{save_name} 保存成功！特征总形状: {feats_tensor.shape}\n")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="把 MobvoiHotwords 特征缓存到 checkpoints/kws/*_cache.pt")
    ap.add_argument("--json-dir", default=os.environ.get("MOBVOI_JSON_DIR"),
                    required=not os.environ.get("MOBVOI_JSON_DIR"),
                    help="p_train.json / n_train.json 等标注文件所在目录（也可设环境变量 MOBVOI_JSON_DIR）")
    ap.add_argument("--audio-dir", default=os.environ.get("MOBVOI_AUDIO_DIR"),
                    required=not os.environ.get("MOBVOI_AUDIO_DIR"),
                    help="wav 音频所在目录（也可设环境变量 MOBVOI_AUDIO_DIR）")
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT_DIR, "checkpoints", "kws"),
                    help="缓存输出目录，默认 <仓库根>/checkpoints/kws")
    ap.add_argument("--workers", type=int, default=8, help="读取 wav 的进程数")
    args = ap.parse_args()

    # 1. 自动创建 checkpoints/kws 文件夹
    CKPT_DIR = args.ckpt_dir
    os.makedirs(CKPT_DIR, exist_ok=True)

    JSON_DIR = args.json_dir
    AUDIO_DIR = args.audio_dir

    print(f"特征口径: htk mel / n_mels={N_MELS} / power=2 / 逐样本减最大值 / 下限 -{TOP_DB:g}dB")
    print(f"标注目录: {JSON_DIR}\n音频目录: {AUDIO_DIR}\n输出目录: {CKPT_DIR}\n")

    train_jsons = [os.path.join(JSON_DIR, "p_train.json"), os.path.join(JSON_DIR, "n_train.json")]
    val_jsons   = [os.path.join(JSON_DIR, "p_dev.json"), os.path.join(JSON_DIR, "n_dev.json")]
    test_jsons  = [os.path.join(JSON_DIR, "p_test.json"), os.path.join(JSON_DIR, "n_test.json")]

    # 2. 将缓存保存到 CKPT_DIR 下（特征口径一变，这三份旧缓存全部作废，这里直接覆盖）
    extract_and_save(train_jsons, os.path.join(CKPT_DIR, "train_cache.pt"), AUDIO_DIR, args.workers)
    extract_and_save(val_jsons, os.path.join(CKPT_DIR, "val_cache.pt"), AUDIO_DIR, args.workers)
    extract_and_save(test_jsons, os.path.join(CKPT_DIR, "test_cache.pt"), AUDIO_DIR, args.workers)
    print(f"全部缓存完成！文件已保存至 {CKPT_DIR}")
