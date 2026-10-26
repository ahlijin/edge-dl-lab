import os
import sys

# 将项目根目录加入 sys.path，解决 import 报错
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from projects.kws.model import DSCNN

def main():
    device = torch.device("cpu")
    print(f"当前使用设备: {device} (纯 CPU 优化模式)")
    torch.set_num_threads(20) 

    CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints", "kws")

    print("加载特征缓存中...")
    train_data = torch.load(os.path.join(CKPT_DIR, "train_cache.pt"))
    val_data = torch.load(os.path.join(CKPT_DIR, "val_cache.pt"))
    test_data = torch.load(os.path.join(CKPT_DIR, "test_cache.pt"))

    # 训练集：负样本太多，按 1:2 比例强行砍掉多余的负样本
    labels = train_data["labels"]
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
    
    keep_neg_num = min(len(pos_idx) * 2, len(neg_idx))
    perm = torch.randperm(len(neg_idx))
    keep_neg_idx = neg_idx[perm[:keep_neg_num]]
    
    all_train_idx = torch.cat([pos_idx, keep_neg_idx])
    all_train_idx = all_train_idx[torch.randperm(len(all_train_idx))] # 打乱
    
    train_dataset = TensorDataset(
        train_data["feats"][all_train_idx], 
        train_data["labels"][all_train_idx]
    )

    # 验证集：也随机抽样 5000 条，加快每轮验证的速度
    val_idx = torch.randperm(len(val_data["labels"]))[:5000]
    val_dataset = TensorDataset(val_data["feats"][val_idx], val_data["labels"][val_idx])
    
    # 测试集：保留全量，用于最终评估
    test_dataset = TensorDataset(test_data["feats"], test_data["labels"])

    print(f"精简后 -> 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")

    # ==========================================
    # 2. 构建 DataLoader
    # ==========================================
    BATCH_SIZE = 256
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 3. 初始化模型与优化器
    # ==========================================
    model = DSCNN(n_classes=2).to(device)
    # 开启 CPU 专属的 channels_last 加速
    model = model.to(memory_format=torch.channels_last)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    EPOCHS = 3 # 从 10 改为 3，Loss 早就降下去了，多跑无益
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    # ==========================================
    # 4. 训练循环
    # ==========================================
    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for feat, label in pbar:
            # 缓存的是 float16，转成 float32 计算，并应用 channels_last
            feat = feat.float().contiguous(memory_format=torch.channels_last)
            label = label.long()
            
            optimizer.zero_grad()
            logits = model(feat)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feat, label in val_loader:
                feat = feat.float().contiguous(memory_format=torch.channels_last)
                label = label.long()
                
                logits = model(feat)
                loss = criterion(logits, label)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 2. 保存模型到 checkpoints/kws/
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_wakeword.pth"))
            print(f"  --> 发现更好的模型，已保存至 {CKPT_DIR}/best_wakeword.pth")

    # ==========================================
    # 5. 最终测试与阈值评估
    # ==========================================
    print("\n开始在测试集上评估...")
    model.load_state_dict(torch.load("best_wakeword.pth"))
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for feat, label in tqdm(test_loader, desc="Testing"):
            feat = feat.float().contiguous(memory_format=torch.channels_last)
            
            logits = model(feat)
            probs = torch.softmax(logits, dim=1)[:, 1] # 取类1（唤醒词）的概率
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print("\n--- 不同阈值下的评估结果 ---")
    for threshold in [0.5, 0.7, 0.9]:
        preds = (all_probs > threshold).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        
        recall = tp / (tp + fn + 1e-8)  # 召回率（唤醒率）
        far = fp / (fp + tn + 1e-8)     # 误唤醒率
        print(f"阈值 {threshold}: 召回率 (Recall) = {recall:.4f}, 误唤醒率 (FAR) = {far:.4f}")

if __name__ == "__main__":
    main()