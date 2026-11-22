import torch

def generate_anchors(feature_map_size, scales, ratios, stride, device="cpu"):
    """
    生成某层特征图上的锚框（以中心点坐标和宽高表示）
    """
    H, W = feature_map_size
    # 生成网络中心点(cx, cy)
    shift_x = torch.arange(0, W) * stride + stride / 2.0
    shift_y = torch.arange(0, H) * stride + stride / 2.0
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')

    # 此时shape都是(H, W)， 展开并堆叠 -> (H*W, 2)
    centers = torch.stack([shift_x.flatten(), shift_y.flatten()], dim=1).to(device)

    # 生成不同尺度和比例的基础框(w, h)
    # 例如 scales=[32, 64], ratios=[0.5, 1.0, 2.0]
    anchors_wh = []
    for scale in scales:
        for ratio in ratios:
            w = scale * torch.sqrt(torch.tensor(1.0 / ratio))
            h = scale * torch.sqrt(torch.tensor(1.0 * ratio))
            anchors_wh.append([w, h])
    anchors_wh = torch.tensor(anchors_wh, dtype=torch.float32).to(device)

    # 组合中心点和宽高 -> （H*W*A, 4)
    # 结果形状为(cx, cy, w, h)
    num_anchors = anchors_wh.shape[0]
    centers = centers.unsqueeze(1).repeat(1, num_anchors, 1) # (H*W, A, 2)
    wh = anchors_wh.unsqueeze(0).repeat(centers.shape[0], 1, 1) # (H*W, A, 2)

    anchors = torch.cat([centers, wh], dim=2).view(-1, 4)
    return anchors

def compute_iou(boxes1, boxes2):
    """
    计算两组框的IoU（交并比）
    boxes1: (N, 4) 格式 (cx, cy, w, h)
    boxes2: (M, 4) 格式 (cx, cy, w, h)
    """

    # 将(cx, cy, w, h)转换为(xmin, ymin, xmax, ymax)
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2

    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

    # 求交集区域 (利用 PyTorch 的广播机制，相当于 C++ 里的双重 for 循环)
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 求并集面积
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    return inter_area / torch.clamp(union_area, min=1e-6)