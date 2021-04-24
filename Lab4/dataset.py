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
        print(f'> Found {len(self.img_name)} images...')

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        '''something you should implement here'''
        '''
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'

           step2. Get the ground truth label from self.label

           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 

                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 

                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]

            step4. Return processed image and label
        '''
        img = Image.open(os.path.join(self.root, f'{self.img_name[index]}.jpeg'))
        label = self.label[index]
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224)
        ])
        img = trans(img)

        return img, label
