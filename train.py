import os
import torch
from torch import nn, optim
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import logging
from pycocotools.coco import COCO
from model import DA_model
import torchvision


def train_one_epoch(model, optimizer, source_loader, target_loader, epoch):
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    pbar = tqdm(zip(source_loader, target_loader), total=min(len(source_loader), len(target_loader)))
    for i, ((source_images, source_labels), (target_images)) in enumerate(pbar):

        source_images = list(image.to('cuda', non_blocking=True) for image in source_images) # list of [C, H, W]
        source_labels = [{k: v.to('cuda', non_blocking=True) for k, v in t.items()} for t in source_labels]
        target_images = list(image.to('cuda', non_blocking=True) for image in target_images) # list of [C, H, W]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=False):
            loss_dict = model(source_images, source_labels, target_images)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix({'loss': losses.item()})


@torch.no_grad()
def validation(model, data_loader):
    metric = MeanAveragePrecision()
    model.eval()

    for images, targets in tqdm(data_loader):
        images = list(image.to('cuda', non_blocking=True) for image in images)
        predictions = model(images)
        predictions = ...  # postprocess: modify format to meet metric's requirements
        metric.update(predictions, targets)
    
    result = metric.compute()
    return result['map_50']


def build_dataloader():

    def collate_fn(batch):
        ...
        # TODO
        # depends on your code
        # return tuple(zip(*batch))

    # TODO
    from dataset import SourceDataset, TargetDataset, get_transform
    source_dataset = SourceDataset()
    target_dataset = TargetDataset()
    val_dataset = ...

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)

    return source_loader, target_loader, val_loader


model = DA_model(n_classes, load_source_model=False)
model = model.to('cuda')


# TODO
optimizer = ...
scheduler = ...
num_epochs = 50

best_epoch = 0
source_loader, target_loader, val_loader = build_dataloader()
best_map = map_50 = validation(model, val_loader)
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, source_loader, target_loader, epoch)
    scheduler.step()

    map_50 = validation(model, val_loader)
    if map_50 > best_map:
        best_map = map_50
        best_epoch = epoch
        torch.save(model.state_dict(), ...)
