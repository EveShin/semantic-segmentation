import torch.nn as nn
import torch.nn.functional as F
import config
from aspp import ASPP
from backbone import ResNet50

class DeepLabv3(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(DeepLabv3, self).__init__()

        self.backbone = ResNet50(pretrained=True)
        self.aspp = ASPP(in_channels=2048, out_channels=256) # layer4 출력 채널 : 2048
        self.classifier = nn.Conv2d(256, num_classes, 1)

        bn_momentum = 1 - config.BATCH_NORM_DECAY

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum

    def forward(self, x):
        input_shape = x.shape[2:] # (B, C, "Height, Width")를 저장해서 업샘플링 할 사이즈 보존

        x = self.backbone(x)
        x = self.aspp(x)
        x = self.classifier(x)

        return F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True) # 업샘플링