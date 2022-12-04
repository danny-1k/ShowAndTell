import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from utils import process_text, read_img
from tokenizer import Tokenizer

train_transforms = transforms.Compose([
  transforms.Rezize((244, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
  transforms.Rezize((244, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Flickr(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()
        if train:
          self.df = pd.read_csv('../data/flickr30k/train.csv')
        else:
          self.df = pd.read_csv('../data/flickr30k/test.csv')

        self.transforms = train_transforms if train else test_transforms

        self.tokenizer = Tokenizer('../tokens/flickr')
        self.tokenier.load()
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        img = train_transforms(
          read_img(os.path.join('../data/flickr30k/flickr30k_images', item['image_name']))
        )
        label = process_text(self.tokenizer, item['comment'])

        return img, label