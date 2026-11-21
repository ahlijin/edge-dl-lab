import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

from dataset import HomeObjectsDataset
from model import TinyDetector
from anchor_utils import generate_anchors

def nms(boxes, scores, iou_threshold=0.45):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    
    # 将 (cx, cy, w, h) 转为 (x1, y1, x2, y2) 方便计算 IoU
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)
    
    # 按置信度降序排列的索引，强制展平为 1D
    order = scores.argsort(descending=True).flatten()
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item()) # 只有1个框时，直接用 item() 提取标量
            break
            
        i = order[0].item()
        keep.append(i)
        
        # 计算该框与剩余所有框的交集
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        inter_w = torch.clamp(xx2 - xx1, min=0)
        inter_h = torch.clamp(yy2 - yy1, min=0)
        inter_area = inter_w * inter_h
        
        # 计算 IoU
        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-6)
        
        # 保留 IoU 小于阈值的框（剔除重叠度高的）
        inds = (iou <= iou_threshold).nonzero().flatten()
        if inds.numel() == 0:
            break
        order = order[inds + 1]  # +1 是因为 order[0] 是我们刚取出的
        
    return torch.tensor(keep, dtype=torch.long)

def decode_predictions(reg_pred, obj_pred, cls_pred, anchors):
    """将网络的预测值解码为真实的坐标"""
    # 1. 解码偏移量
    ax, ay, aw, ah = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    tx, ty, tw, th = reg_pred[:, 0], reg_pred[:, 1], reg_pred[:, 2], reg_pred[:, 3]
    
    pred_w = aw * torch.exp(tw)
    pred_h = ah * torch.exp(th)
    pred_cx = ax + tx * aw
    pred_cy = ay + ty * ah
    
    pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
    
    # 2. 处理置信度和分类
    # 网络输出的是 logits (未激活的分数)，需要过 Sigmoid 和 Softmax
    obj_scores = torch.sigmoid(obj_pred).squeeze(-1) # (N,)
    cls_probs = torch.softmax(cls_pred, dim=-1) # (N, 12)
    
    # 获取每个框最可能属于的类别和对应概率
    max_cls_prob, max_cls_idx = cls_probs.max(dim=-1)
    
    # 最终得分 = 置信度 * 类别概率
    final_scores = obj_scores * max_cls_prob
    
    return pred_boxes, final_scores, max_cls_idx

if __name__ == "__main__":
    device = torch.device("cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 加载模型
    # model = TinyDetector(num_classes=12, num_anchors=6).to(device)
    model = TinyDetector(num_classes=12, num_anchors=12).to(device)
    weights_path = os.path.join(current_dir, "detector_weights.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # 开启推理模式
    print("模型加载成功！")

    # 2. 加载数据
    data_dir = os.path.join(current_dir, "data", "homeobjects-3K")
    dataset = HomeObjectsDataset(data_dir, split="val", input_size=(320, 320))
    img, gt_boxes, gt_classes = dataset[random.randint(0, len(dataset)-1)]
    
    # 3. 前向推理
    with torch.no_grad():
        img_input = img.unsqueeze(0).to(device)
        reg_pred, cls_pred, obj_pred = model(img_input)
        
        # 剥离 Batch 维度
        reg_pred = reg_pred[0]
        cls_pred = cls_pred[0]
        obj_pred = obj_pred[0]
        
        # 生成 Anchors
        # anchors = generate_anchors((40, 40), scales=[32, 64], ratios=[0.5, 1.0, 2.0], stride=8).to(device)
        anchors = generate_anchors((40, 40), scales=[32, 64, 128, 256], ratios=[0.5, 1.0, 2.0], stride=8)
        
        # 解码预测结果
        pred_boxes, final_scores, pred_cls = decode_predictions(reg_pred, obj_pred, cls_pred, anchors)
        
        # 4. 过滤低分框 (阈值设为 0.3，可根据实际效果调整)
        print(f"过滤前的最高得分: {final_scores.max().item():.4f}")
        score_thresh = 0.2
        keep_mask = final_scores > score_thresh
        pred_boxes = pred_boxes[keep_mask]
        final_scores = final_scores[keep_mask]
        pred_cls = pred_cls[keep_mask]
        
        # 5. NMS 去重
        keep_indices = nms(pred_boxes, final_scores, iou_threshold=0.45)
        pred_boxes = pred_boxes[keep_indices]
        final_scores = final_scores[keep_indices]
        pred_cls = pred_cls[keep_indices]
        
    print(f"过滤后保留了 {pred_boxes.shape[0]} 个预测框。")

    # 6. 可视化
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(img.permute(1, 2, 0).numpy())
    
    # 画真实框（红框）
    for gt in gt_boxes:
        cx, cy, w, h = gt.numpy()
        rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
    # 画预测框（绿框 + 分数）
    for box, score, cls_id in zip(pred_boxes, final_scores, pred_cls):
        cx, cy, w, h = box.cpu().numpy()
        rect = patches.Rectangle((cx - w/2, cy - h/2), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(cx - w/2, cy - h/2, f"{cls_id.item()}:{score.item():.2f}", color='white', fontsize=10, bbox=dict(facecolor='green', alpha=0.5))

    plt.title("Red: Ground Truth, Green: Predictions")
    plt.show()