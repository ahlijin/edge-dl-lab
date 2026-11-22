import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))

class TinyDetector(nn.Module):
    def __init__(self, num_classes=12, num_anchors=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # 简化的Backbone（假设输入320x320，下采样stride=8，输出特征图40x40）
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 128, stride=2),
            DepthwiseSeparableBlock(128, 128, stride=1)
        )

        # 回归头（预测4个偏移量）
        self.reg_head = nn.Conv2d(128, num_anchors * 4, kernel_size=3, padding=1)
        # 分类头（预测 类别数 个概率）
        self.cls_head = nn.Conv2d(128, num_anchors * num_classes, kernel_size=3, padding=1)
        # 置信度头（预测 是否有物体）
        self.obj_head = nn.Conv2d(128, num_anchors * 1, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.backbone(x)

        # 预测并展平： (B, A*C, H, W) -> (B, H*W*A, C)
        reg_pred = self.reg_head(features).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4)
        cls_pred = self.cls_head(features).permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.num_classes)
        obj_pred = self.obj_head(features).permute(0, 2, 3, 1).reshape(x.shape[0], -1, 1)

        return reg_pred, cls_pred, obj_pred
    