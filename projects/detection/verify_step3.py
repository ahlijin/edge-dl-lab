import torch
import os
from anchor_utils import generate_anchors
from dataset import HomeObjectsDataset
from target_assigner import AnchorAssigner
from model import TinyDetector
from loss import DetectionLoss
import random

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载一条数据
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", "homeobjects-3K")
    dataset = HomeObjectsDataset(data_dir, split="val", input_size=(320, 320))
    img, gt_boxes, gt_classes = dataset[random.randint(0, 403)]

    # 增加batch维度
    img = img.unsqueeze(0).to(device)
    gt_boxes = gt_boxes.to(device)
    gt_classes = gt_classes.to(device)

    # 2. 生成Anchors与标签
    anchors = generate_anchors((40, 40), scales=[32, 64], ratios=[0.5, 1.0, 2.0], stride=8).to(device)
    assigner = AnchorAssigner(pos_iou_thresh=0.5, neg_iou_thresh=0.3)
    objectness, box_targets, class_targets = assigner(anchors, gt_boxes, gt_classes)

    # 增加batch维度
    objectness = objectness.unsqueeze(0).to(device)
    box_targets = box_targets.unsqueeze(0).to(device)
    class_targets = class_targets.unsqueeze(0).to(device)

    # 3. 实例化模型和loss
    model = TinyDetector(num_classes=12, num_anchors=6).to(device)
    criterion = DetectionLoss()

    # 4. 前向传播
    reg_pred, cls_pred, obj_pred = model(img)
    print(f"网络输出形状 - 回归：{reg_pred.shape}, 分类：{cls_pred.shape}, 置信度：{obj_pred.shape}")

    # 5. 计算损失
    total_loss, obj_loss, cls_loss, reg_loss = criterion(
        reg_pred, cls_pred, obj_pred,
        objectness, box_targets, class_targets
    )

    print("\n--- 初始损失（随机权重） ---")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"  - Objectness Loss: {obj_loss.item():.4f}")
    print(f"  - Classification Loss: {cls_loss.item():.4f}")
    print(f"  - Regression Loss: {reg_loss.item():.4f}")