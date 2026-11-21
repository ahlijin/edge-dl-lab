import torch
import os
import time
from torch.utils.data import DataLoader

torch.set_num_threads(10) 

from anchor_utils import generate_anchors
from dataset import HomeObjectsDataset
from target_assigner import AnchorAssigner
from model import TinyDetector
from loss import DetectionLoss

def collate_fn(batch):
    """处理不同图片 GT 数量不一致的问题，把它们打包成列表"""
    images = []
    gts = []
    classes = []
    for img, gt_boxes, gt_classes in batch:
        images.append(img)
        gts.append(gt_boxes)
        classes.append(gt_classes)

    # 图片维度是一致的，可以直接 stack 成一个 Tensor
    # GT 数量不一致，必须保留为 list
    return torch.stack(images), gts, classes

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 数据加载
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", "homeobjects-3K")

    batch_size = 4
    train_dataset = HomeObjectsDataset(data_dir, split="train", input_size=(320, 320))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    # 2. 实例化模型，loss和优化器
    # model = TinyDetector(num_classes=12, num_anchors=6).to(device)
    model = TinyDetector(num_classes=12, num_anchors=12).to(device)
    criterion = DetectionLoss()

    # AdamW是目前最稳妥的优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # 余弦退火学习率：前期大步走，后期小步微调
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    # 3. 锚框只需要生成一次（因为参数固定了）
    # anchors = generate_anchors((40, 40), scales=[32, 64], ratios=[0.5, 1.0, 2.0], stride=8).to(device)
    anchors = generate_anchors((40, 40), scales=[32, 64, 128, 256], ratios=[0.5, 1.0, 2.0], stride=8)
    assiger = AnchorAssigner(pos_iou_thresh=0.5, neg_iou_thresh=0.3)

    epochs = 25

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time  = time.time()

        for batch_idx, (images, gt_boxes_list, gt_classes_list) in enumerate(train_loader):
            images = images.to(device)
            # --- 4. 动态打标签 (针对 Batch 里的每一张图) ---
            batch_objectness = []
            batch_box_targets = []
            batch_class_targets = []

            for i in range(len(images)):
                gt_b = gt_boxes_list[i].to(device)
                gt_c = gt_classes_list[i].to(device)
                obj, box_t, cls_t = assiger(anchors, gt_b, gt_c)
                batch_objectness.append(obj)
                batch_box_targets.append(box_t)
                batch_class_targets.append(cls_t)

            # 堆叠成 (B, N) 和 (B, N, 4)
            objectness = torch.stack(batch_objectness)
            box_targets = torch.stack(batch_box_targets)
            class_targets = torch.stack(batch_class_targets)

            # --- 5. 前向传播 ---
            reg_pred, cls_pred, obj_pred = model(images)

            # --- 6. 计算损失 ---
            total_loss, obj_l, cls_l, reg_l = criterion(
                reg_pred, cls_pred, obj_pred, 
                objectness, box_targets, class_targets
            )

            # --- 7. 反向传播与优化 ---
            optimizer.zero_grad()  # 清空上一轮的梯度
            total_loss.backward()  # 计算当前梯度
            # 梯度裁剪：防止偶尔出现的极端异常值把网络参数“炸”飞
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()       # 更新权重

            epoch_loss += total_loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {total_loss.item():.4f} "
                      f"(Obj: {obj_l.item():.3f}, Cls: {cls_l.item():.3f}, Reg: {reg_l.item():.3f})")

        scheduler.step() # 更新学习率

        avg_loss = epoch_loss / len(train_loader)
        print(f"=== Epoch [{epoch+1}/{epochs}] 结束 | 平均 Loss: {avg_loss:.4f} | 耗时: {time.time()-start_time:.1f}s ===\n")

        # 8. 保存模型权重
    torch.save(model.state_dict(), os.path.join(current_dir, "detector_weights.pth"))
    print("训练完成，模型已保存！")

if __name__ == "__main__":
    train()