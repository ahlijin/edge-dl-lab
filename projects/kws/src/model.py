import torch
import torch.nn as nn

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))

class DSCNN(nn.Module):
    def __init__(self, n_classes=12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(64, 64),
            DepthwiseSeparableBlock(64, 64),
            DepthwiseSeparableBlock(64, 160, stride=2),
            DepthwiseSeparableBlock(160, 160),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(160, n_classes)

    def forward(self, x):
         # x: (B, 1, 64, T)
         x = self.stem(x)
         x = self.blocks(x)
         x = self.pool(x).flatten(1)    # (B, 160)
         x = self.dropout(x)
         return self.fc(x)              # (B, n_classes)