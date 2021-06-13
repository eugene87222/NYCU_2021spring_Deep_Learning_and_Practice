# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms

from cglow import CondGlowModel
from evaluator import evaluation_model
from CLEVR_dataset import CLEVRDataset
from CelebA_dataset import get_CelebA_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def split(img_list, label_list, attrs, target_attr):
    idx, = np.where(attrs==target_attr)
    if len(idx) == 0:
        return []
    else:
        idx = idx[0]
        pos_img = img_list[label_list[:,idx]==1]
        pos_label = label_list[label_list[:,idx]==1]
        neg_img = img_list[label_list[:,idx]==-1]
        neg_label = label_list[label_list[:,idx]==-1]
        return [pos_img, pos_label, neg_img, neg_label]


def compute_average_z(root_dir, img_list, label_list, trans, model):
    avg_z, avg_cond = None, None
    for idx in range(img_list.shape[0]):
        img = Image.open(os.path.join(root_dir, 'CelebA-HQ-img', img_list[idx])).convert('RGB')
        img = trans(img).unsqueeze(dim=0).to(device)
        cond = torch.Tensor(label_list[idx]).unsqueeze(dim=0).to(device)
        with torch.no_grad():
            z = model(x=img, cond=cond, reverse=False)
        if avg_z is None:
            avg_z = z
            avg_cond = cond
        else:
            avg_z = avg_z + z
            avg_cond = avg_cond + cond
    return avg_z/img_list.shape[0], avg_cond/img_list.shape[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    # model
    parser.add_argument('--input_sz', type=int, nargs='+', default=[3, 64, 64])
    parser.add_argument('--cond_sz', type=int) # CLEVR 24, CelebA 40
    parser.add_argument('--cond_fc_fts', type=int, default=64)
    parser.add_argument('--affine_conv_chs', type=int, default=256)
    parser.add_argument('--flow_depth', type=int, default=8)
    parser.add_argument('--num_levels', type=int, default=5)
    parser.add_argument('--img_h', type=int, default=64)
    parser.add_argument('--img_w', type=int, default=64)

    parser.add_argument('--dataset', type=str, help='CLEVR, CelebA')
    parser.add_argument(
        '--task', type=int,
        help='1: conditional face generation, 2: linear interpolation, 3: attribute manipulation')
    parser.add_argument('--cpt_path', type=str)
    parser.add_argument('--output_dir', type=str, default='results')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = CondGlowModel(args).to(device)
    model.load_state_dict(torch.load(args.cpt_path)['model'])
    model.eval()

    if args.dataset == 'CLEVR':
        dataset = CLEVRDataset('../../DLP_Lab7_dataset/task_1', None, mode='test')
        loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=8)
        eval_model = evaluation_model()
        avg_acc = 0
        gen_images = None
        with torch.no_grad():
            for _, conds in enumerate(loader):
                conds = conds.to(device)
                fake_images, _ = model(x=None, cond=conds, reverse=True)
                if gen_images is None:
                    gen_images = fake_images
                else:
                    gen_images = torch.vstack((gen_images, fake_images))
                acc = eval_model.eval(fake_images, conds)
                avg_acc += acc
        avg_acc /= len(loader)
        print(f'Accuracy: {avg_acc:.4f}')
        save_image(gen_images, os.path.join(args.result_dir, 'CLEVR.png'), nrow=8)
    elif args.dataset == 'CelebA':
        root_dir = '../../DLP_Lab7_dataset/task_2'
        img_list, label_list, attrs = get_CelebA_data(root_dir)
        trans = transforms.Compose([
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor()
        ])
        indices = np.arange(img_list.shape[0]-1)
        if args.task == 1:
            gen_images = []
            with torch.no_grad():
                for i in range(3):
                    cond = random.choice(label_list)
                    cond = torch.Tensor(cond).unsqueeze(dim=0).to(device)
                    for _ in range(4):
                        fake_image, _ = model(x=None, cond=cond, reverse=True)
                        gen_images.append(fake_image[0])
            save_image(gen_images, os.path.join(args.result_dir, 'CelebA_task1.png'), nrow=4)
        elif args.task == 2:
            idx_left, idx_right = random.choices(indices, k=2)
            img_left = Image.open(os.path.join(root_dir, 'CelebA-HQ-img', img_list[idx_left])).convert('RGB')
            img_left = trans(img_left).unsqueeze(dim=0).to(device)
            cond_left = torch.Tensor(label_list[idx_left]).unsqueeze(dim=0).to(device)
            img_right = Image.open(os.path.join(root_dir, 'CelebA-HQ-img', img_list[idx_right])).convert('RGB')
            img_right = trans(img_right).unsqueeze(dim=0).to(device)
            cond_right = torch.Tensor(label_list[idx_right]).unsqueeze(dim=0).to(device)
            with torch.no_grad():
                z_left = model(x=img_left, cond=cond_left, reverse=False)
                z_right = model(x=img_right, cond=cond_right, reverse=False)
            gen_images = []
            with torch.no_grad():
                for alpha in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                    z = (1-alpha)*z_left + alpha*z_right
                    cond = (1-alpha)*cond_left + alpha*cond_right
                    fake_image = model(x=z, cond=cond, reverse=True)
                    gen_images.append(fake_image[0])
            save_image(gen_images, os.path.join(args.result_dir, 'CelebA_task2.png'), nrow=6)
        elif args.task == 3:
            for target_attr in ['smiling', 'chubby']:
                result = split(img_list, label_list, attrs, target_attr)
                if len(result) == 0:
                    print(f'<{target_attr}> not exists')
                else:
                    pos_img, pos_label, neg_img, neg_label = result
                    z_pos, cond_pos = compute_average_z(root_dir, pos_img, pos_label, trans, model)
                    z_neg, cond_neg = compute_average_z(root_dir, neg_img, neg_label, trans, model)
                    idx = random.choice(indices)
                    img = Image.open(os.path.join(root_dir, 'CelebA-HQ-img', img_list[idx])).convert('RGB')
                    img = trans(img).unsqueeze(dim=0).to(device)
                    cond = torch.Tensor(label_list[idx]).unsqueeze(dim=0).to(device)
                    with torch.no_grad():
                        z = model(x=img, cond=cond, reverse=False)
                    attr_vec = z_pos - z_neg
                    gen_images = []
                    for alpha in [-0.5, -0.25, 0, 0.25 ,0.5]:
                        fake_image = model(x=z+alpha*attr_vec, cond=cond, reverse=True)
                        gen_images.append(fake_image[0])
                    save_image(gen_images, os.path.join(args.result_dir, 'CelebA_task3.png'), nrow=6)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
