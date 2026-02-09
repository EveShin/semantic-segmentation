# Semantic Segmentation 실현: DeepLabv3 및 PSPNet

본 저장소는 PyTorch를 사용하여 구현한 DeepLabv3 및 PSPNet 모델을 포함하고 있습니다. 두 모델 모두 ResNet-50을 백본(Backbone)으로 사용하며, 밀집된 특징 추출을 위해 확장된 컨볼루션(Dilated Convolution) 구조를 채택했습니다.

## 주요 구현 사항

### 1. 공통 백본: 수정된 ResNet-50

- **다일레이션 전략**: ResNet의 `layer3`와 `layer4`에서 표준 컨볼루션을 확장된 컨볼루션(Atrous Convolution)으로 교체하여 수용 영역을 유지하면서 고해상도 특징 맵(Output Stride 8/16)을 추출합니다.
- **사전 학습 가중치**: ImageNet으로 학습된 가중치를 로드하여 학습 수렴 속도를 높였습니다.

### 2. DeepLabv3

- **ASPP (Atrous Spatial Pyramid Pooling)**: 다중 비율(6, 12, 18)을 가진 병렬 Atrous 컨볼루션을 통해 다양한 크기의 문맥 정보를 포착합니다.
- **전역 문맥 정보**: 전역 평균 풀링(Global Average Pooling)을 통해 이미지 레벨의 특징을 결합합니다.

### 3. PSPNet (Pyramid Scene Parsing Network)

- **PPM (Pyramid Pooling Module)**: 네 가지 서로 다른 스케일(1×1, 2×2, 3×3, 6×6)의 피라미드 풀링을 통해 전역 문맥 정보를 집계합니다.
- **보조 손실 (Auxiliary Loss)**: 학습 중 기울기 소실 문제를 방지하고 성능을 향상시키기 위해 백본의 `layer3` 끝에 보조 헤드를 추가했습니다.

## 프로젝트 구조

- `backbone.py`: 다일레이션 및 멀티 그리드 설정이 포함된 공통 ResNet-50
- `aspp.py` / `ppm.py`: 각 모델의 핵심 모듈인 ASPP와 PPM
- `deeplab.py` / `pspnet.py`: 최종 모델 조립 클래스
- `aug.py`: 데이터 증강(Mirroring, Scaling, Rotation, Gaussian Blur, Random Crop)
- `train.py`: 학습, 검증, 체크포인트 저장을 수행하는 통합 스크립트
- `dataloader.py`: PASCAL VOC 데이터셋 로더
- `metrics.py`: 혼동 행렬 기반의 Pixel Accuracy 및 mIoU 계산

## 설치 및 환경 설정
```bash
pip install torch torchvision numpy pillow
```

## 사용 방법

### 학습
```bash
python train.py --model deeplab --epochs 50 --batch-size 8
```

### 평가
```bash
python eval.py --model pspnet --checkpoint path/to/checkpoint.pth
```

## 참고 문헌

- Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587.
- Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid scene parsing network. In CVPR.

## 라이선스

MIT License
