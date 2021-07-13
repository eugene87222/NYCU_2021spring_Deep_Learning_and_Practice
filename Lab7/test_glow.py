# -*- coding: utf-8 -*-
import os
import pickle
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


def compute_average_z(target_attr, has_attr, root_dir, trans, model, batch_size):
    dataset = CelebADataset(root_dir, trans, target_attr=target_attr, has_attr=has_attr)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)
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

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--n_bits', default=5, type=int)

    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--temp', type=float, default=0.6)
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

    target_attrs = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
        'Wearing_Necktie', 'Young'
    ]

    # compute attribute vectors of each attribute
    # for target_attr in target_attrs:
    #     z_pos = compute_average_z(target_attr, True, root_dir, trans, model, args.batch_size)
    #     z_neg = compute_average_z(target_attr, False, root_dir, trans, model, args.batch_size)
    #     attr_z = []
    #     for z_idx in range(len(z_pos)):
    #         attr_z.append((z_pos[z_idx]-z_neg[z_idx]).cpu().numpy())
    #     with open(f'attr/{target_attr.lower()}.attr', 'wb') as fp:
    #         pickle.dump(attr_z, fp)
    #     print(f'attr/{target_attr.lower()}.attr')

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_sz, args.flow_depth, args.num_levels)
    for shape in z_shapes:
        z_uniform = torch.randn(args.num_samples, *shape) * args.temp
        z_normal = torch.normal(torch.zeros(args.num_samples, *shape), torch.ones(args.num_samples, *shape)) * args.temp
        z_sample.append(torch.vstack((z_uniform, z_normal)).to(device))
    with torch.no_grad():
        gen_images = model.reverse(z_sample, reconstruct=False)
    save_image(
        gen_images+0.5,
        os.path.join(args.output_dir, 'sample.png'),
        nrow=4)

    # interpolation
    n = args.interpolate_step
    k = 10
    target_idx = random.choices(indices, k=k)
    target_zs, zs = [], []
    for idx in target_idx:
        img = read_image(os.path.join(root_dir, 'CelebA-HQ-img', img_list[idx]), trans)
        with torch.no_grad():
            _, _, z = model.forward(img)
        target_zs.append(z)
    zs_l, zs_r = target_zs[:k//2], target_zs[k//2:]
    for z_idx in range(len(zs_l[0])):
        z = []
        for z_pair in zip(zs_l, zs_r):
            for i in range(n):
                new_z = (n-i)*z_pair[0][z_idx]/n + i*z_pair[1][z_idx]/n
                z.append(new_z)
        zs.append(torch.cat(z))
    with torch.no_grad():
        gen_images = model.reverse(zs, reconstruct=True)
    save_image(
        gen_images+0.5,
        os.path.join(args.output_dir, 'task2_interpolate.png'),
        nrow=n)

    # attribute manipulation
    n = args.manipulation_step
    idx = random.choice(indices)
    img = read_image(os.path.join(root_dir, 'CelebA-HQ-img', img_list[idx]), trans)
    with torch.no_grad():
        _, _, z_img = model.forward(img)
    for target_attr in ['Blond_Hair', 'Goatee', 'Smiling']:
        with open(f'attr/{target_attr.lower()}.attr', 'rb') as fp:
            attr = pickle.load(fp)
        zs = []
        for z_idx in range(len(attr)):
            attr_vec = torch.Tensor(attr[z_idx]).to(device)
            z = []
            for i in range(-n, 0):
                new_z = z_img[z_idx] + attr_vec*i*0.4
                z.append(new_z)
            z.append(z_img[z_idx])
            for i in range(1, n+1):
                new_z = z_img[z_idx] + attr_vec*i*0.4
                z.append(new_z)
            zs.append(torch.cat(z))
        with torch.no_grad():
            gen_images = model.reverse(zs, reconstruct=True)
        save_image(
            gen_images+0.5,
            os.path.join(args.output_dir, f'task3_manipulation_{target_attr.lower()}.png'),
            nrow=2*n+1)
