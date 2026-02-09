import os
import torch
import torch.nn as nn
import config

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) # 1x1 conv 줄이고
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=dilation, dilation=dilation, bias=False) # 3x3 conv 추출하고
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False) # 1x1 conv 4배 expansion
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x # 원본

        out = self.conv1(x) # 1x1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1
        out = self.bn3(out)

        if self.downsample is not None: # 입출력 모양 다르면 맞춰주기
            # downsample = None : 모양이 같다
            identity = self.downsample(x)

        out = out + identity # Skip Connection
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, multi_grid=config.MULTI_GRID):
        super(ResNet, self).__init__()

        # 7x7 하나가 아닌 3x3 세 개로 수용 영역 유지 비선형성 증가
        self.in_channels = 128
        self.conv1 = nn.Conv2d(3, 64, 3,
                                   2, 1, bias = False) # 2
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3,
                                   1, 1,bias = False) # 2
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3,
                                   1, 1, bias = False) # 2
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 4

        # Layer 1~4 생성 ex) layers=[3, 4, 6, 3] 첫번째 섹션은 bottleneck 3개를 만들자
        self.layer1 = self._make_layer(block, 64, layers[0], 1, 1) # 4
        self.layer2 = self._make_layer(block, 128, layers[1], 2, 1) # 8

        # Layer 3, 4는 stride 1로 해상도 유지, dilation 적용해서 수용 영역 늘림
        self.layer3 = self._make_layer(block, 256, layers[2], 1, 2) # os = 16 : stride = 2, dilation = 1     os = 8 : stride = 1, dilation = 2
        self.layer4 = self._make_layer(block, 512, layers[3], 1, 4, multi_grid) # os = 16 : dilation = 2     os = 8 : dilation = 4

    def _make_layer(self, block, out_channel, blocks, stride = 1, dilation = 1, multi_grid=None):
        # 연속된 Bottleneck block 생성

        downsample = None

        # stride가 1이 아니거나(해상도 바뀜), 입출력 채널 다를 때 downsample 변수에 모양 맞추는 레이어 담기
        if stride != 1 or self.in_channels != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channel * block.expansion,
                          kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channel * block.expansion)
            ) # 블록 클래스에 넘겨줄 레이어 생성

        layers = [] # 바구니 생성

        base_dil = dilation
        dil = base_dil * multi_grid[0] if multi_grid else base_dil
        layers.append(block(self.in_channels, out_channel, stride, dil, downsample))
        self.in_channels = out_channel * block.expansion

        for i in range(1, blocks):
            dil = base_dil * multi_grid[i] if multi_grid and i < len(multi_grid) else base_dil
            layers.append(block(self.in_channels, out_channel, dilation=dil))

        return nn.Sequential(*layers) # 한 스테이지 완성


    def forward(self, x):
        x = self.conv1(x) # 3x3
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x) # 3x3
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x) # 3x3
        x = self.bn3(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x) # bottleneck : 1x1 -> 3x3 -> 1x1
        x = self.layer2(x)  # bottleneck : 1x1 -> 3x3 -> 1x1

        x = self.layer3(x) # bottleneck : 1x1 -> 3x3 -> 1x1
        x = self.layer4(x) # bottleneck : 1x1 -> 3x3 -> 1x1

        return x


def ResNet50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3])

    if pretrained:
        weight_path = os.path.join(config.MODEL_DIR, "resnet50_v2.pth")
        state_dict = torch.load(weight_path, map_location=config.DEVICE)
        model.load_state_dict(state_dict, strict=False)

    return model