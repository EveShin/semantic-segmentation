import torch
import os

NUM_CLASSES = 21

WEIGHT_DECAY = 0.0001
LEARNING_RATE = 0.01
MOMENTUM = 0.9

MAX_ITER = 60000
BATCH_SIZE = 8
AUX_WEIGHT = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CLASS_NAME = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
              'sofa', 'train', 'tvmonitor']

VOC_COLORMAP = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128],
                [128, 128, 0], [128, 128, 128], [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0], [0, 192, 128], [128, 64, 0]] # BGR

TRAIN_MODE = 'train_aug'
VAL_MODE = 'val'
TEST_MODE = 'test'

ROOT_DIR = r"/dataset\train"
TRAIN_LIST_DIR = os.path.join(ROOT_DIR, "train_list")
VAL_LIST_DIR = os.path.join(ROOT_DIR, "split_list")
MODEL_DIR = os.path.join("../pretrained")
SAVE_DIR = "./model/checkpoints"
LOG_DIR = "./model/logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
