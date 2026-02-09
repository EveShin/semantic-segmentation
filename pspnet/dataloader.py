import os
import config
from torch.utils.data import Dataset
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform
        self.mode = mode

        list_dir = config.TRAIN_LIST_DIR if mode == config.TRAIN_MODE else config.VAL_LIST_DIR
        list_path = os.path.join(list_dir, f"{mode}.txt")

        self.image_dir = os.path.join(self.root, "JPEGImages")
        self.label_dir = os.path.join(self.root, "SegmentationClassAug")

        with open(list_path, 'r') as f:
            self.file_names = f.read().splitlines() # 텍스트 파일의 파일명 리스트에 저장

        print(f"{mode} dataset loaded: {len(self.file_names)} images")

    def __getitem__(self, idx):
        img_id = self.file_names[idx] # 해당 인덱스의 파일 id
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        label_path = os.path.join(self.label_dir, f"{img_id}.png")

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform is not None:
            image, label = self.transform(image, label)

        if self.mode == 'train_aug':
            return image, label
        else:
            return image, label, img_id

    def __len__(self):
        return len(self.file_names)