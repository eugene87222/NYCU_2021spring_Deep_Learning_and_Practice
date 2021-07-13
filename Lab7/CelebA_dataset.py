# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


def get_CelebA_data(root_dir, target_attr=None, has_attr=True):
    if target_attr is not None:
        target_attr = target_attr.lower()
    img_list = sorted(os.listdir(os.path.join(root_dir, 'CelebA-HQ-img')), key=lambda t: int(t.split('/')[-1][:-4]))
    label_list = []
    f = open(os.path.join(root_dir, 'CelebA-HQ-attribute-anno.txt'), 'r')
    num_imgs = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    img_list = np.array(img_list)
    label_list = np.array(label_list)
    attrs = np.array([attr.lower() for attr in attrs])
    if target_attr is not None:
        idx, = np.where(attrs==target_attr)
        if len(idx) == 0:
            return []
        else:
            idx = idx[0]
            if has_attr is True:
                target_img = img_list[label_list[:,idx]==1]
                target_label = label_list[label_list[:,idx]==1]
            else:
                target_img = img_list[label_list[:,idx]==-1]
                target_label = label_list[label_list[:,idx]==-1]
            return target_img, target_label, attrs
    else:
        return img_list, label_list, attrs


class CelebADataset(Dataset):
    def __init__(self, root_dir, trans, cond=True, target_attr=None, has_attr=True):
        self.root_dir = root_dir
        assert os.path.isdir(self.root_dir), f'{self.root_dir} is not a valid directory'
        self.trans = trans
        self.cond = cond
        try:
            self.img_list, self.label_list, self.attrs = get_CelebA_data(self.root_dir, target_attr, has_attr)
        except:
            raise ValueError(f'<{target_attr}> not found')
        self.num_classes = 40
        print(f'> Found {len(self.img_list)} images...')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, 'CelebA-HQ-img', self.img_list[index])).convert('RGB')
        img = self.trans(img)
        if self.cond:
            cond = self.label_list[index]
            return img, torch.Tensor(cond)
        else:
            return img
