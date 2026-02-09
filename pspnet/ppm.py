import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.Module):
    def __init__(self, in_channels, bin_sizes=[1, 2, 3, 6]): # 1x1, 2x2, 3x3, 6x6
        super(PPM, self).__init__()

        self.stages = nn.ModuleList()
        out_channels = in_channels // len(bin_sizes) # 나중에 합치면 동일해짐

        for size in bin_sizes: # 서로 다른 Sequential 레이어를 만들어서 리스트에 추가
            self.stages.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size), # 해당 bin 크기로 풀링 (Adaptive : 입력 크기에 상관없이 출력 size 고정)
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), # 1x1 Conv로 채널 축소 (계산량 감소, 특징 요약)
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        h, w = x.size(2), x.size(3)  # x: 백본에서 넘어온 원본 특징 맵

        out = [x] # 원본 특징 맵(x) 리스트에 미리 담음

        for stage in self.stages:
            pooled_feat = stage(x) # 풀링 + 1x1 Conv (작은 특징 추출)
            upsampled_feat = F.interpolate(pooled_feat, size=(h, w), mode='bilinear', align_corners=True) # 업샘플링
            out.append(upsampled_feat) # 확대된 특징 맵을 리스트에 추가

        return torch.cat(out, dim=1) # 모든 특징 맵을 채널 방향(dim=1)으로 Concat