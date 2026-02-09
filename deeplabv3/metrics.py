import numpy as np
import config

class Metrics:
    def __init__(self, num_class=config.NUM_CLASSES):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

    # 전체 중 맞춘 비율
    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum() # confusion_matrix 대각 행렬 합 / 전체 합
        return acc

    # 각 클래스별 겹치는 비율
    def miou(self):
        intersection = np.diag(self.confusion_matrix) # confusion matrix의 대각행렬
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection # confusion matrix의 row(행) + column(열) - 교집합(중복)
        iou = intersection / union # 나눠주기
        miou = np.nanmean(iou) # 모든 클래스의 IoU 값들 평균
        return iou, miou

    # confusion_matrix 만들기
    def add_batch(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class) # 클래스 범위 내에 있는 거
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask] # num_class * gt + pre -> flatten
        count = np.bincount(label, minlength=self.num_class**2) # (gt-pre)가 몇 개인지
        confusion_matrix = count.reshape(self.num_class, self.num_class) # 다시 2차원
        self.confusion_matrix += confusion_matrix # 업데이트

    # confusion matrix를 0으로 만듦
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))