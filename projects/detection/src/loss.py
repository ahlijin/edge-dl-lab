import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_cls = nn.CrossEntropyLoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, reg_pred, cls_pred, obj_pred, objectness, box_targets, class_targets):
        # 1. 全部展平
        reg_pred = reg_pred.view(-1, 4)
        box_targets = box_targets.view(-1, 4)
        cls_pred = cls_pred.view(-1, cls_pred.shape[-1])
        class_targets = class_targets.view(-1)
        obj_pred = obj_pred.view(-1)
        objectness = objectness.view(-1)

        # 2. 生成掩码
        pos_mask = (objectness == 1.0)
        valid_mask = (objectness != -1.0)
        num_pos = pos_mask.sum().item()
        num_valid = valid_mask.sum().item()

        # 3. 目标置信度损失
        obj_loss_all = self.bce_obj(obj_pred, objectness)
        obj_loss = (obj_loss_all * valid_mask).sum() / (num_valid + 1e-6)

        if num_pos > 0:
            # 4. 分类损失 (关键修改在这里)
            # CrossEntropyLoss 要求标签在 [0, num_classes-1] 范围内
            # 所以把所有非正样本（-1）的标签临时替换成 0
            safe_class_targets = class_targets.clone()
            safe_class_targets[~pos_mask] = 0  # 非正样本设为 0（合法类别）
            
            cls_loss_all = self.ce_cls(cls_pred, safe_class_targets)
            cls_loss = (cls_loss_all * pos_mask).sum() / num_pos

            # 5. 回归损失
            reg_loss_all = self.smooth_l1(reg_pred, box_targets).sum(dim=-1)
            reg_loss = (reg_loss_all * pos_mask).sum() / num_pos
        else:
            cls_loss = torch.tensor(0.0, device=reg_pred.device)
            reg_loss = torch.tensor(0.0, device=reg_pred.device)

        # 6. 加权求和
        total_loss = 1.0 * obj_loss + 1.0 * cls_loss + 1.0 * reg_loss
        return total_loss, obj_loss, cls_loss, reg_loss