# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_data(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = get_data(mode)
        self.mode = mode
        transform = [
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0),
            transforms.Resize(size=224)
        ]
        self.transform = transforms.RandomOrder(transform)
        self.to_tensor = transforms.ToTensor()
        print(f'> Found {len(self.img_name)} images...')

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, f'{self.img_name[index]}.jpeg'))
        label = self.label[index]
        img = self.transform(img)
        img = self.to_tensor(img)
        return img, label
