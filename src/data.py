import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from utils import process_text, read_img

train_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Flickr(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()
        if train:
          self.df = pd.read_csv('../data/flickr30k/train.csv', sep='| ')
        else:
          self.df = pd.read_csv('../data/flickr30k/test.csv', sep='| ')

        self.transforms = train_transforms if train else test_transforms
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df[index]
        img = train_transforms(os.path.join('../data/flickr30k/flickr30k_images', item['image_name']))
        label = process_text(item['comment'])

        return img, label