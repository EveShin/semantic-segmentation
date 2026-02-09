import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class RandomMirror(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = F.hflip(image)
            label = F.hflip(label)
        return image, label


class RandomScale(object):
    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, image, label):
        scale = random.uniform(*self.scale_range)
        new_size = [int(x * scale) for x in image.size[::-1]]
        return F.resize(image, new_size), F.resize(label, new_size, interpolation=Image.NEAREST)

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        w, h = image.size
        th, tw = self.size

        if w < tw or h < th:
            pad_h = max(th - h, 0)
            pad_w = max(tw - w, 0)

            image = F.pad(image, (0, 0, pad_w, pad_h), fill=0)
            label = F.pad(label, (0, 0, pad_w, pad_h), fill=255)
            w, h = image.size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return F.crop(image, i, j, th, tw), F.crop(label, i, j, th, tw)


class ToTensorAndNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        label = torch.from_numpy(np.array(label)).long()
        return image, label

