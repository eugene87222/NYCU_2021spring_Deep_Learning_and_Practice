# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms

from glow import Glow
from main_glow import calc_z_shapes
from CelebA_dataset import get_CelebA_data, CelebADataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_image(path, trans):
    img = Image.open(path).convert('RGB')
    img = trans(img).unsqueeze(dim=0).to(device)
    img = img - 0.5
    # img = img * 255
    # if n_bits < 8:
    #     img = torch.floor(img/2**(8-n_bits))
    # img = img/n_bins - 0.5
    # img = img + torch.rand_like(img)/n_bins
    return img


def compute_average_z(target_attr, has_attr, root_dir, trans, model):
    dataset = CelebADataset(root_dir, trans, target_attr=target_attr, has_attr=has_attr)
    loader = DataLoader(dataset, batch_size=48, num_workers=8)
    avg_z = None
    pbar_loader = tqdm(loader)
    pbar_loader.set_description(f'attr: {target_attr}, has: {has_attr}')
    with torch.no_grad():
        for images, conds in pbar_loader:
            images = images.to(device)
            _, _, zs = model.forward(images)
            if avg_z is None:
                avg_z = []
                for z_idx in range(len(zs)):
                    avg_z.append(zs[z_idx].sum(dim=0)/len(dataset))
            else:
                for z_idx in range(len(zs)):
                    avg_z[z_idx] = avg_z[z_idx] + zs[z_idx].sum(dim=0)/len(dataset)
    return avg_z


if __name__ == '__main__':
    parser = ArgumentParser()
    # model
    parser.add_argument('--img_sz', type=int, default=64)
    parser.add_argument('--affine_conv_chs', type=int, default=256)
    parser.add_argument('--flow_depth', type=int, default=32)
    parser.add_argument('--num_levels', type=int, default=4)

    parser.add_argument('--n_bits', default=5, type=int)

    parser.add_argument(
        '--task', type=int,
        help='2: linear interpolation, 3: attribute manipulation')
    parser.add_argument('--interpolate_step', type=int, default=8)
    parser.add_argument('--manipulation_step', type=int, default=3)

    parser.add_argument('--cpt_path', type=str)
    parser.add_argument('--output_dir', type=str, default='output')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    root_dir = '../../DLP_Lab7_dataset/task_2'
    img_list, label_list, attrs = get_CelebA_data(root_dir)
    trans = transforms.Compose([
        transforms.Resize((args.img_sz, args.img_sz)),
        transforms.ToTensor()
    ])
    indices = np.arange(img_list.shape[0]-1)
    n_bins = 2 ** args.n_bits

    model = Glow(
        in_chs=3,
        flow_depth=args.flow_depth,
        num_levels=args.num_levels,
        affine_conv_chs=args.affine_conv_chs,
        actnorm_inited=True).to(device)
    model.load_state_dict(torch.load(args.cpt_path))
    model.eval()

    # interpolation
    if args.task == 2:
        n = args.interpolate_step
        target_idx = random.choices(indices, k=6)
        target_zs, zs = [], []
        for idx in target_idx:
            img = read_image(os.path.join(root_dir, 'CelebA-HQ-img', img_list[idx]), trans)
            _, _, z = model.forward(img)
            target_zs.append(z)
        zs_l, zs_r = target_zs[:3], target_zs[3:]
        for z_idx in range(len(zs_l[0])):
            z = []
            for z_pair in zip(zs_l, zs_r):
                for i in range(n):
                    new_z = (n-i)*z_pair[0][z_idx]/n + i*z_pair[1][z_idx]/n
                    z.append(new_z)
            zs.append(torch.cat(z))
        gen_images = model.reverse(zs, reconstruct=True)
        save_image(
            gen_images+0.5,
            os.path.join(args.outout_dir, 'task2_interpolate.png'),
            nrow=n)

    # attribute manipulation
    elif args.task == 3:
        n = args.manipulation_step
        idx = random.choice(indices)
        img = read_image(os.path.join(root_dir, 'CelebA-HQ-img', img_list[idx]), trans)
        _, _, z_img = model.forward(img)
        # arched_eyebrows attractive bags_under_eyes bald big_lips big_nose
        # black_hair blond_hair chubby eyeglasses goatee heavy_makeup smiling
        # wavy_hair wearing_hat young
        for target_attr in ['chubby', 'young', 'arched_eyebrows', 'bald', 'black_hair', 'blond_hair']:
            z_pos = compute_average_z(target_attr, True, root_dir, trans, model)
            z_neg = compute_average_z(target_attr, False, root_dir, trans, model)
            zs = []
            for z_idx in range(len(z_pos)):
                attr_vec = z_pos[z_idx] - z_neg[z_idx]
                z = []
                for i in range(-n, 0):
                    new_z = z_img[z_idx] + attr_vec*i/2
                    z.append(new_z)
                z.append(z_img[z_idx])
                for i in range(1, n+1):
                    new_z = z_img[z_idx] + attr_vec*i/2
                    z.append(new_z)
                zs.append(torch.cat(z))
            gen_images = model.reverse(zs, reconstruct=True)
            save_image(
                gen_images+0.5,
                os.path.join(args.output_dir, f'task3_manipulation_{target_attr}.png'),
                nrow=2*n+1)
    else:
        raise ValueError(f'Unknown task: task {args.task}')
