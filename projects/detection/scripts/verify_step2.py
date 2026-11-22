import torch
import os
import sys
import random

# 本文件在 scripts/ 下，模块在 ../src：把它加进 import 搜索路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # projects/detection
sys.path.insert(0, os.path.join(current_dir, "src"))

from anchor_utils import generate_anchors
from dataset import HomeObjectsDataset
from target_assigner import AnchorAssigner
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":
    data_dir = os.path.join(current_dir, "data", "homeobjects-3K")
    
    dataset = HomeObjectsDataset(data_dir, split="val", input_size=(320, 320))
    img, gt_boxes, gt_classes = dataset[random.randint(0, 403)]
    
    # 1. 生成锚框
    # 假设特征图 40x40, 步长 8, 尺度 [32, 64], 比例 [0.5, 1, 2]
    anchors = generate_anchors((40, 40), scales=[32, 64], ratios=[0.5, 1.0, 2.0], stride=8)
    
    # 2. 标签分配
    assigner = AnchorAssigner(pos_iou_thresh=0.5, neg_iou_thresh=0.3)
    objectness, box_targets, class_targets = assigner(anchors, gt_boxes, gt_classes)
    
    # 3. 统计正负样本
    num_pos = (objectness == 1.0).sum().item()
    num_neg = (objectness == 0.0).sum().item()
    num_ign = (objectness == -1.0).sum().item()
    print(f"正样本数量: {num_pos}, 负样本数量: {num_neg}, 忽略样本数量: {num_ign}")
    
    # 4. 验证：把正样本的 anchor 加上偏移量，还原成预测框，看是否和红框完美重合
    print("\n开始验证还原精度...")

    pos_indices = torch.nonzero(objectness == 1.0).flatten() 
    
    fig, ax = plt.subplots(1)
    ax.imshow(img.permute(1, 2, 0).numpy())
    
    for idx in pos_indices:
        anchor = anchors[idx]
        target = box_targets[idx]
        
        # 解码（Inverse Encode）
        ax_cx, ax_cy, ax_w, ax_h = anchor
        tx, ty, tw, th = target
        
        # 用 exp 还原出预测框的真实宽高
        pred_w = ax_w * torch.exp(tw)
        pred_h = ax_h * torch.exp(th)
        pred_cx = ax_cx + tx * ax_w
        pred_cy = ax_cy + ty * ax_h
        
        # 画出还原后的框（蓝色实线）
        rect = patches.Rectangle(
            (pred_cx - pred_w/2, pred_cy - pred_h/2), 
            pred_w, pred_h,
            linewidth=2.4, edgecolor='blue', facecolor='none',
            linestyle=(0, (2, 6)), zorder=5, label='Decoded Box'
        )
        ax.add_patch(rect)
        
    # 画出 GT 框（红色）
    for gt in gt_boxes:
        cx, cy, w, h = gt.numpy()
        rect = patches.Rectangle(
            (cx - w/2, cy - h/2),
            w, h, 
            linewidth=2.4, edgecolor='r', facecolor='none', 
            linestyle=(0, (5, 5)), zorder=4)
        ax.add_patch(rect)

    plt.title("Red: GT, Blue: Decoded from Target")
    plt.show()