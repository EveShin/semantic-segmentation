import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
import numpy as np
from torch.utils.data import DataLoader
from deeplab import DeepLabv3
from aug import Compose, RandomMirror, RandomScale, RandomCrop, ToTensorAndNormalize
from dataloader import VOCDataset
from metrics import Metrics
from poly import Poly_Scheduler
from PIL import Image


def train():
    print(f"DEVICE: {config.DEVICE}")

    log_file = open(os.path.join(config.LOG_DIR, 'log.txt'), 'a+') # txt 파일
    log_file_class = open(os.path.join(config.LOG_DIR, 'class_miou.txt'), 'a+')  # txt 파일

    #mean = config.MEAN
    #std = config.STD

    print("Loading pretrained model...")
    model = DeepLabv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    print("Loading data...")
    train_transform = Compose([
        RandomMirror(),
        RandomScale(scale_range=(0.5, 2.0)),
        RandomCrop(size=(513, 513)),
        ToTensorAndNormalize(mean=config.MEAN, std=config.STD)
    ]) # 모두 수행

    train_dataset = VOCDataset(root=config.ROOT_DIR, mode=config.TRAIN_MODE, transform=train_transform) # train dataset 생성
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True) # train dataloader 생성

    val_transform = Compose([ToTensorAndNormalize(mean=config.MEAN, std=config.STD)]) # 정규화만 수행
    val_dataset = VOCDataset(root=config.ROOT_DIR, mode=config.VAL_MODE, transform=val_transform) # val dataset
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2) # val dataloader

    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY) # optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255) # loss
    evaluator = Metrics(config.NUM_CLASSES) # 지표 계산기

    curr_iter = 0
    best_miou = 0.0

    resume_path = os.path.join(config.SAVE_DIR, "checkpoint_7000.pth")

    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=config.DEVICE)

        model.load_state_dict(checkpoint['model_state_dict'])  # 가중치 복구
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 옵티마이저 복구
        curr_iter = checkpoint['iteration'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)

        torch.cuda.empty_cache()
        print(f"Restarting from iteration {curr_iter}")
    else:
        print("No checkpoint found.")

    # 학습 루프 시작
    while curr_iter < config.MAX_ITER:
        model.train() # train mode
        for images, targets in train_loader:
            if curr_iter >= config.MAX_ITER: break

            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)
            lr = Poly_Scheduler(optimizer, config.LEARNING_RATE, curr_iter, config.MAX_ITER) # 스케줄러

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad() # gradient 초기화

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward() # backpropagate,  weight update
            optimizer.step()

            # 로깅, 저장
            if curr_iter % 1000 == 0:
                preds = torch.argmax(outputs, dim=1) # 가장 확률 높은 클래스 선택
                evaluator.add_batch(targets.cpu().numpy(), preds.cpu().numpy())
                acc = evaluator.pixel_accuracy()
                miou = evaluator.miou()[1]
                evaluator.reset()

                log_msg = (f"it: {curr_iter}      loss: {loss.item():.4f}   "
                           f"lr: {lr:.8f}   pixel acc: {acc:.2f}   miou: {miou:.4f}\n")
                log_file.write(log_msg)
                log_file.flush()

                checkpoint = {
                    'iteration': curr_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_miou
                }
                ckpt_path = os.path.join(config.SAVE_DIR, f"checkpoint_{curr_iter}.pth")
                torch.save(checkpoint, ckpt_path)
                print(f"Checkpoint saved at iteration {curr_iter}")

                torch.cuda.empty_cache()

            curr_iter += 1

            # validate
            if curr_iter % 5000 == 0:
                iou, miou = validate(model, val_loader, config.DEVICE, evaluator, curr_iter)

                if miou > best_miou:
                    best_miou = miou
                    save_path = os.path.join(config.SAVE_DIR, "best_model.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"Iter {curr_iter}: Best mIoU {miou:.4f}")

                log_file.write(f"it: {curr_iter}   val mIoU: {miou:.4f}\n")
                log_file_class.write(f"it: {curr_iter}\n")

                for i, class_iou in enumerate(iou):
                    class_name = config.CLASS_NAME[i]
                    log_file_class.write(f"{class_name:15s}     iou: {class_iou:.4f}\n")

                log_file.flush()
                log_file_class.flush()

                model.train() # 끝나면 다시 train mode

    log_file.close()
    log_file_class.close()

def validate(model, val_loader, device, evaluator, curr_iter):
    model.eval()
    evaluator.reset()

    scales = config.MULTI_SCALES

    colormap = []

    for color_BGR in config.VOC_COLORMAP:
        color_RGB = color_BGR[::-1]
        colormap.extend(color_RGB)

    if len(colormap) < 768:
        colormap.extend([0] * (768 - len(colormap)))

    pred_save_dir = os.path.join(config.SAVE_DIR, "predictions", f"iter_{curr_iter}")
    os.makedirs(pred_save_dir, exist_ok=True)
    log_file_single_miou = open(os.path.join(pred_save_dir, f"{curr_iter}_single.txt"), 'a+')

    with torch.no_grad():
        for i, (images, targets, image_id) in enumerate(val_loader):
            images = images.to(device)

            h, w = images.shape[2:]

            final_probs = torch.zeros((1, config.NUM_CLASSES, h, w)).to(device)

            for s in scales:
                resize_h, resize_w = int(h * s), int(w * s)
                input_img = F.interpolate(images, size=(resize_h, resize_w), mode='bilinear', align_corners=True)

                output = model(input_img)
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)

                final_probs += torch.softmax(output, dim=1)

                flip_img = torch.flip(input_img, dims=[3])
                flip_output = model(flip_img)
                flip_output = torch.flip(flip_output, dims=[3])
                flip_output = F.interpolate(flip_output, size=(h, w), mode='bilinear', align_corners=True)
                final_probs += torch.softmax(flip_output, dim=1)

            num_inputs = len(scales) * 2
            probs_avg = final_probs / num_inputs

            preds = torch.argmax(probs_avg, dim=1)

            evaluator.add_batch(targets.numpy(), preds.cpu().numpy())

            if i < 100:
                temp_evaluator = Metrics(config.NUM_CLASSES)
                temp_evaluator.add_batch(targets.numpy(), preds.cpu().numpy())
                single_miou = temp_evaluator.miou()[1]

                log_file_single_miou.write(f"id: {image_id[0]}   Single mIoU: {single_miou:.4f}\n")

                pred_mask = preds[0].cpu().numpy().astype(np.uint8)
                img = Image.fromarray(pred_mask)
                img.putpalette(colormap)
                img.save(os.path.join(pred_save_dir, f"{curr_iter}_pred_{image_id[0]}_{single_miou:.4f}.png"))

        iou, miou = evaluator.miou()
        log_file_single_miou.close()

    return iou, miou

'''def validate(model, val_loader, device, evaluator, curr_iter):
    model.eval() # evaluation
    evaluator.reset()

    colormap = []

    for color_BGR in config.VOC_COLORMAP:
        color_RGB = color_BGR[::-1]
        colormap.extend(color_RGB)

    if len(colormap) < 768:
        colormap.extend([0] * (768 - len(colormap)))

    pred_save_dir = os.path.join(config.SAVE_DIR, "predictions", f"iter_{curr_iter}") # 이미지 저장 경로
    os.makedirs(pred_save_dir, exist_ok=True)
    log_file_single_miou = open(os.path.join(pred_save_dir, f"{curr_iter}_single.txt"), 'a+')  # txt 파일

    with torch.no_grad():
        for i, (images, targets, image_id) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            evaluator.add_batch(targets.cpu().numpy(), preds.cpu().numpy())

            if i < 100:
                temp_evaluator = Metrics(config.NUM_CLASSES)
                temp_evaluator.add_batch(targets.cpu().numpy(), preds.cpu().numpy())

                single_miou = temp_evaluator.miou()[1]

                log_file_single_miou.write(f"id: {image_id[0]}   Single mIoU: {single_miou:.4f}\n")

                pred_mask = preds[0].cpu().numpy().astype(np.uint8)
                img = Image.fromarray(pred_mask)
                img.putpalette(colormap)
                img.save(os.path.join(pred_save_dir, f"{curr_iter}_pred_{image_id[0]}_{single_miou:.4f}.png"))

        iou, miou = evaluator.miou()

        log_file_single_miou.close()

    return iou, miou'''

if __name__ == "__main__":
    log_file_class = open(os.path.join(config.LOG_DIR, 'class_miou_os8_multi.txt'), 'a+')

    model = DeepLabv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    checkpoint_path = os.path.join(config.SAVE_DIR, "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    print(f"Model Loaded")

    val_transform = Compose([ToTensorAndNormalize(mean=config.MEAN, std=config.STD)])
    val_dataset = VOCDataset(root=config.ROOT_DIR, mode=config.VAL_MODE, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    evaluator = Metrics(config.NUM_CLASSES)

    iou, miou = validate(model, val_loader, config.DEVICE, evaluator, curr_iter="OS8_multi_Validation_2")

    log_file_class.write(f"\nOS=8 + Multi-Scale + Flip Evaluation\n")
    log_file_class.write(f"Final mIoU: {miou:.4f}\n")

    for i, class_iou in enumerate(iou):
        class_name = config.CLASS_NAME[i]
        log_file_class.write(f"{class_name:15s}     iou: {class_iou:.4f}\n")

    log_file_class.flush()

    print(f"Evaluation Finished. mIoU: {miou:.4f}")
