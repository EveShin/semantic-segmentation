import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False), # os = 16: 6      os = 8: 12
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=24, dilation=24, bias=False), # os = 16: 12     os = 8: 24
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=36, dilation=36, bias=False), # os = 16: 18     os = 8: 36
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential( # 분류 헤드에 넣기 전에 연산량을 줄이기 위해 압축
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)

        x5= F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True) # 원본 크기로 업샘플링

        out = torch.cat([x1, x2, x3, x4, x5], dim=1) # concat
        return self.bottleneck(out)

