import torch
from anchor_utils import compute_iou


class AnchorAssigner:
    def __init__(self, pos_iou_thresh=0.5, neg_iou_thresh=0.3):
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh

    def __call__(self, anchors, gt_boxes, gt_classes):
        num_anchors = anchors.shape[0]
        objectness = torch.zeros(num_anchors, dtype=torch.float32)      # (N,)
        box_targets = torch.zeros((num_anchors, 4), dtype=torch.float32)
        class_targets = torch.full((num_anchors,), -1, dtype=torch.int64)

        if gt_boxes.shape[0] == 0:
            return objectness, box_targets, class_targets

        ious = compute_iou(anchors, gt_boxes)                       # (N, M)
        max_iou_per_anchor, gt_idx_per_anchor = ious.max(dim=1)      # 每个 anchor 的最佳 GT

        # 1) 忽略区（先做，后面正样本会覆盖它）
        objectness[(max_iou_per_anchor >= self.neg_iou_thresh) &
                   (max_iou_per_anchor < self.pos_iou_thresh)] = -1.0

        # 2) 阈值正样本（向量化）
        pos_mask = max_iou_per_anchor >= self.pos_iou_thresh
        objectness[pos_mask] = 1.0
        class_targets[pos_mask] = gt_classes[gt_idx_per_anchor[pos_mask]]
        box_targets[pos_mask] = self.encode(anchors[pos_mask],
                                            gt_boxes[gt_idx_per_anchor[pos_mask]])

        # 3) 兜底：每个 GT 至少一个正样本（最后做，保证不被覆盖）
        #    修复：原来是无条件把"该 GT 最像的 anchor"改成正样本，如果这个 anchor
        #    已经属于别的 GT（两个 GT 抢同一个 anchor），后面的写入会把前面的覆盖掉，
        #    被抢的那个 GT 就一个正样本都没有了。现在只给"还没有正样本"的 GT 兜底，
        #    并且只在还没被占用的 anchor 里挑 IoU 最高的。
        pos_count_per_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.int64)
        for g_idx in gt_idx_per_anchor[pos_mask]:
            pos_count_per_gt[g_idx] += 1

        for i in range(gt_boxes.shape[0]):
            if pos_count_per_gt[i] > 0:
                continue

            iou_this_gt = ious[:, i].clone()
            iou_this_gt[objectness == 1.0] = -1.0       # 已被占用的 anchor 排除掉
            a_idx = int(iou_this_gt.argmax())

            objectness[a_idx] = 1.0
            class_targets[a_idx] = gt_classes[i]
            box_targets[a_idx] = self.encode(anchors[a_idx:a_idx + 1], gt_boxes[i:i + 1])
            pos_count_per_gt[i] = 1

        return objectness, box_targets, class_targets

    @staticmethod
    def encode(anchor, gt_box):
        ax, ay, aw, ah = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        gx, gy, gw, gh = gt_box[:, 0], gt_box[:, 1], gt_box[:, 2], gt_box[:, 3]
        return torch.stack([
            (gx - ax) / aw,
            (gy - ay) / ah,
            torch.log(torch.clamp(gw / aw, min=1e-6)),
            torch.log(torch.clamp(gh / ah, min=1e-6)),
        ], dim=1)
