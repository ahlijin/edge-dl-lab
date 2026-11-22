import torch 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import random

# 本文件在 scripts/ 下，模块在 ../src：把它加进 import 搜索路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # projects/detection
sys.path.insert(0, os.path.join(current_dir, "src"))

from anchor_utils import generate_anchors, compute_iou
from dataset import HomeObjectsDataset

def visualize(image, gt_boxes, matched_boxes):
    """可视化真实框（红色）和匹配到的锚框（绿色）"""
    fig, ax = plt.subplots(1)
    img_np = image.permute(1, 2, 0).numpy()
    ax.imshow(img_np)

    # GT
    for box in gt_boxes:
        cx, cy, w, h = box.numpy()
        rect = patches.Rectangle((cx-w/2, cy-h/2), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # 匹配到的Anchor
    for box in matched_boxes:
        cx, cy, w, h = box.numpy()
        rect = patches.Rectangle((cx-w/2, cy-h/2), w, h, linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    # 测试1 生成锚框
    anchors = generate_anchors((40, 40), scales=[32, 64], ratios=[0.5, 1.0, 2.0], stride=8)
    print(f"特征图（40x40）生成的Anchor数量：{anchors.shape[0]}") # 40*40*6=9600

    # 测试2 加载数据集并匹配锚框
    dataset = HomeObjectsDataset(os.path.join(current_dir, "data", "homeobjects-3K"), split="val", input_size=(320, 320))
    img, gt_boxes, classes = dataset[random.randint(0, 403)]
    print(f"图像shape: {img.shape}, GT框数量：{gt_boxes.shape[0]}")

    if gt_boxes.shape[0] > 0:
        # 计算所有Anchor和GT框的IoU
        ious = compute_iou(anchors, gt_boxes)
        max_iou, max_idx = ious.max(dim=0)
        matched_anchors = anchors[max_idx]

        print(f"最高IoU：{max_iou.max().item():.4f}")
        print("正在渲染可视化结果...")
        visualize(img, gt_boxes, matched_anchors)

