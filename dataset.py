import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import torch
import copy
from PIL import Image
from glob import glob
import json


def get_transform(*args, **kwargs):
    # Define the transformations to be applied
    if kwargs['train']:
        transforms = [
            T.Resize((800, 800)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.RandomPerspective(),
            T.RandomAffine(30),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    else:
        transform = [
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # Combine the transformations into a single transform
    transform = T.Compose(transforms)
    
    return transform


class SourceDataset(Dataset):

    def __init__(self, root: str, split: str = "train", transform=None, *args, **kwargs) -> None:
        super().__init__()

        self.root = root
        self.split = split
        self.transform = transform

        ann_file = os.path.join(self.root, f"org/{self.split}.coco.json")
        with open(ann_file, "r") as f:
            self.coco = json.load(f)
        
        self.ids = list(sorted(self.coco["images"].keys()))

    def _load_image(self, index: int):
        ...
        img_info = self.coco["images"][self.ids[index]]
        ima_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(ima_path).convert("RGB")
        return img
        # return image

    def _load_target(self, index: int):
        img_id = self.coco["images"][self.ids[index]]
        anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))

        boxes = [ann["bbox"] + [ann["category_id"]] for ann in anns]

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor([ann["category_id"] for ann in anns], dtype=torch.int64)

        return target
        # return target

    def __getitem__(self, index: int):
        
        image = self._load_image(index)
        target = copy.deepcopy(self._load_target(index))
        
        boxes = [t['bbox'] + [t['category_id']] for t in target]
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes)
        
        image = transformed['image']
        boxes = transformed['bboxes']
        # xmin, ymin, w, h -> xmin, ymin, xmax, ymax
        new_boxes = []
        for box in boxes:
            xmin =  box[0]
            ymin = box[1]
            xmax = xmin + box[2]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(new_boxes, dtype=torch.float32)
        
        targ = {}
        targ["boxes"] = boxes
        targ["labels"] = torch.tensor([t["category_id"]  for t in target], dtype=torch.int64)
        targ["image_id"] = torch.tensor([t["image_id"]  for t in target])
        targ["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targ["iscrowd"] = torch.tensor([t["iscrowd"]  for t in target], dtype=torch.int64)
        
        # TODO: make sure your image is scaled properly
        return image, targ
        # return image and target

    def __len__(self) -> int:
        return len(self.ids)


class TargetDataset(Dataset):

    def __init__(self, root: str, split: str = "train", transform=None, *args, **kwargs) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        # Load the annotations for the validation split of the target dataset
        if self.split == "val":
            ann_file = os.path.join(self.root, f"fog/{self.split}.coco.json")
            with open(ann_file, "r") as f:
                self.coco = json.load(f)
            self.ids = list(sorted(self.coco["images"].keys()))
        else:
            self.ids = self._load_image_ids(root)

        
    def _load_image_ids(self, root: str):
        # Load the image ids for the training split of the target dataset
        image_ids = []
        for img_path in glob(os.path.join(root, f"fog/{self.split}*.jpg")):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            image_ids.append(img_id)
        return image_ids

    def _load_image(self, index: int):
        # Load the image for the specified index
        img_id = self.ids[index]
        img_path = os.path.join(self.root, self.split, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        return img

    def _load_target(self, index: int):
        # Load the annotations for the specified index
        if self.split == "train":
            # During training, return an empty dictionary for the target
            return {}

        elif self.split == "val":
            # During validation, load the annotations for the specified index
            img_id = self.ids[index]
            ann_ids = self.annotations.getAnnIds(imgIds=img_id)
            anns = self.annotations.loadAnns(ann_ids)

            # Convert the bounding boxes to the required format
            boxes = [ann["bbox"] + [ann["category_id"]] for ann in anns]

            # Create the target dictionary
            target = {}
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor([ann["category_id"] for ann in anns], dtype=torch.int64)
            target["image_id"] = torch.tensor([img_id], dtype=torch.int64)
            target["area"] = torch.tensor([ann["area"] for ann in anns], dtype=torch.float32)
            target["iscrowd"] = torch.tensor([ann["iscrowd"] for ann in anns], dtype=torch.int64)

            return target

    def __getitem__(self, index: int):
        # Load the image and target for the specified index
        image = self._load_image(index)
        target = self._load_target(index)

        # Apply the specified transformations to the image and target
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self) -> int:
        # Return the length of the dataset
        return len(self.ids)
