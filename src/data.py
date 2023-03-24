import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CocoDataset(Dataset):
    def __init__(self,  tokenizer, transforms, train: bool = True):
        self.train = train
        self.transforms = transforms
        self.tokenizer  = tokenizer
        if train:
            self.annotations = json.load(
                open("../data/coco/annotations/captions_train2014.json", "r"))
        else:
            self.annotations = json.load(
                open("../data/coco/annotations/captions_val2014.json", "r"))

        self.annotations = self.annotations["annotations"]

    def __getitem__(self, index):
        ann = self.annotations[index]
        image_id = ann["image_id"]
        caption = ann["caption"]
        image_file = "0" * (12-len(str(image_id))) + str(image_id) # image file id is padded with zeros from the left
        image_file = f"../data/coco/{'train2014' if self.train else 'val2014'}/COCO_{'train' if self.train else 'val'}2014_{image_file}.jpg"

        img = Image.open(image_file).convert("RGB") #ensure image is in RGB
        image = self.transforms(img)
        caption = torch.Tensor(self.tokenizer.encode(caption))
        return image, caption

    def __len__(self):
        return len(self.annotations)
