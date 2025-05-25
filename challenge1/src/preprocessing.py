"""
Author: Annam.ai IIT Ropar
Team Name: GeoGenesis
Team Members: Khwaish Yadav, Hemant, Sparsh Patidar, Smarth Tripathi, Sai Pradeep
Leaderboard Rank: 86

This file handles all the preprocessing steps for the Soil Image Classification Challenge.
"""

import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Label encoding
label2idx = {'Alluvial soil': 0, 'Black Soil': 1, 'Clay soil': 2, 'Red soil': 3}
idx2label = {v: k for k, v in label2idx.items()}

# Image transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Custom dataset class
class SoilDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, train=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_dir, row['image_id'])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = label2idx[row['soil_type']]
            return image, label
        else:
            return image, row['image_id']


def preprocessing():
    print("This is the file for preprocessing")
    return 0
