import torch.nn as nn
import torch.nn.functional as F
from backbone import ResNet50
from ppm import PPM

class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()

        # 백본으로 이미지 특징 추출
        self.backbone = ResNet50()

        # PPM으로 다양한 필터를 통해 global context 추출
        self.ppm = PPM(in_channels=2048)

        # 최종 세그멘테이션 맵 생성
        self.main_head = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        # Auxiliary Head : 하위 레이어 학습 도우미
        self.aux_head = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3) # 업샘플링을 위해 가로(w), 세로(h) 크기 저장

        # 백본 (x_aux : 중간 특징맵, x_main : 최종 특징맵)
        x_aux, x_main = self.backbone(x)

        # PPM : global context 결합
        x = self.ppm(x_main)

        # 최종 분류
        x = self.main_head(x)

        # 업샘플링
        main_out = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        # 학습할 때만 Auxiliary Head 연산
        if self.training:
            aux_out = self.aux_head(x_aux)
            aux_out = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=True) # 보조 헤드 결과도 업샘플링
            return main_out, aux_out

        return main_out