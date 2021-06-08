# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


def get_CLEVR_data(root_dir, mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_dir, 'train.json')))
        obj = json.load(open(os.path.join(root_dir, 'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_dir, 'test.json')))
        obj = json.load(open(os.path.join(root_dir, 'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class CLEVRDataset(Dataset):
    def __init__(self, root_dir, trans, cond=True, mode='train'):
        self.root_dir = root_dir
        self.trans = trans
        self.cond = cond
        self.mode = mode
        self.img_list, self.label_list = get_CLEVR_data(root_dir, mode)
        self.num_classes = 24
        if mode == 'train':
            print(f'> Found {len(self.img_list)} images...')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(os.path.join(self.root_dir, 'images', self.img_list[index])).convert('RGB')
            img = self.trans(img)
            if self.cond:
                cond = self.label_list[index]
                return img, torch.Tensor(cond)
            else:
                return img
        else:
            cond = self.label_list[index]
            return torch.Tensor(cond)
