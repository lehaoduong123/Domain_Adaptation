import os
import torch
from torch import nn, optim
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import logging
from pycocotools.coco import COCO
from model import DA_model
import torchvision
import matplotlib.pyplot as plt
from dataset import SourceDataset, TargetDataset, get_transform


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
    
def normal_train_one_epoch(model, optimizer, source_loader, epoch):
    model.train()

    pbar = tqdm(source_loader)
    total_loss = 0.0
    num_batches = len(source_loader)
    for i, (images, targets) in enumerate(pbar):
        images = list(image.to('cuda', non_blocking=True) for image in images)
        targets = [{k: v.to('cuda', non_blocking=True) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix({'loss': losses.item()})
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")


@torch.no_grad()
def validation(model, data_loader):
    metric = MeanAveragePrecision()
    model.eval()

    for images, targets in tqdm(data_loader):
        images = list(image.to('cuda', non_blocking=True) for image in images)
        predictions = model(images)
        predictions = [{k: v.to('cpu') for k, v in t.items()} for t in predictions]
        metric.update(predictions, targets)
    
    result = metric.compute()
    return result['map_50']


def build_dataloader():

    def collate_fn(batch):
        ...
        # TODO
        # depends on your code
        return tuple(zip(*batch))

    # TODO
    from dataset import SourceDataset, TargetDataset, get_transform
    source_dataset = SourceDataset(root="content/hw3_dataset/", split="train", transforms=get_transform("train"))
    target_dataset = TargetDataset(root="content/hw3_dataset/", split="train", transforms=get_transform("train"))
    val_dataset =   TargetDataset(root="content/hw3_dataset/", split="val", transforms=get_transform("val"))
    
    batch_size = 4
    num_workers = 4
    
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers, shuffle=True, collate_fn=collate_fn)

    return source_loader, target_loader, val_loader


model = DA_model(9, load_source_model=False)
model = model.to('cuda')


# TODO
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 50
DA = False

best_epoch = 0
mAP50_values = []
epoch_numbers = []
# if(DA):
for epoch in range(num_epochs):
#         source_loader = SourceDataset(root="content/hw3_dataset/", split="train", transforms=get_transform("train"))
#         train_one_epoch(model, optimizer, source_loader, epoch)
#     else:
    source_loader, target_loader, val_loader = build_dataloader()
    best_map = map_50 = validation(model, val_loader)
    train_one_epoch(model, optimizer, source_loader, target_loader, epoch)
    scheduler.step()

    map_50 = validation(model, val_loader)
    if map_50 > best_map:
        best_map = map_50
        best_epoch = epoch
        torch.save(model.state_dict(), f'model_{epoch}.pth')
    if epoch % 10 ==0:
        mAP50_values.append(map_50)
        epoch_numbers.append(epoch)

plt.plot(epoch_numbers, mAP50_values)
plt.xlabel('epoch')
plt.ylabel('map_50')
plt.title("mAP@50 vs Epoch")
plt.show()

