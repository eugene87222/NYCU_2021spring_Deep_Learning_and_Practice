# -*- coding: utf-8 -*-
import os
from math import log
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from cglow import CondGlow
from CelebA_dataset import CelebADataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample_data(batch_sz, img_sz):
    transform = transforms.Compose([
        transforms.Resize((img_sz, img_sz)),
        transforms.ToTensor()
    ])
    dataset = CelebADataset('../../DLP_Lab7_dataset/task_2', transform, cond=True)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_sz, num_workers=8)
    loader = iter(loader)
    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(dataset, shuffle=True, batch_size=batch_sz, num_workers=8)
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_chs, img_sz, flow_depth, num_levels):
    z_shapes = []
    for i in range(num_levels-1):
        img_sz //= 2
        n_chs *= 2
        z_shapes.append((n_chs, img_sz, img_sz))
    img_sz //= 2
    z_shapes.append((n_chs*4, img_sz, img_sz))
    return z_shapes


def calc_loss(log_p, log_det, img_sz, n_bins):
    n_pixel = img_sz * img_sz * 3
    loss = -log(n_bins) * n_pixel
    loss = loss + log_det + log_p
    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (log_det / (log(2) * n_pixel)).mean()
    )


def train(args, model, optimizer, log_dir, cpt_dir, result_dir):
    logger = SummaryWriter(log_dir)
    os.makedirs(cpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    dataset = iter(sample_data(args.bs, args.img_sz))
    n_bins = 2 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_sz, args.flow_depth, args.num_levels)
    for shape in z_shapes:
        z = torch.randn(args.num_samples, *shape) * args.temp
        z_sample.append(z.to(device))

    pbar_iter = tqdm(range(args.num_iters))
    for i in pbar_iter:
        model.train()
        image = next(dataset)
        image = image.to(device)

        image = image * 255
        if args.n_bits < 8:
            image = torch.floor(image/2**(8-args.n_bits))
        image = image/n_bins - 0.5

        model.zero_grad()
        optimizer.zero_grad()
        log_p, log_det = model.forward(image+torch.rand_like(image)/n_bins)
        log_det = log_det.mean()
        loss, log_p, log_det = calc_loss(log_p, log_det, args.img_sz, n_bins)
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        pbar_iter.set_description(f'Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {lr:.7f}')
        logger.add_scalar('iter/loss', loss.item(), i+1)
        logger.add_scalar('iter/log_p', log_p.item(), i+1)
        logger.add_scalar('iter/log_det', log_det.item(), i+1)
        logger.add_scalar('iter/lr', lr, i+1)

        if (i+1)%args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                save_image(
                    model.reverse(z_sample).cpu().data,
                    os.path.join(result_dir, f'{str(i+1).zfill(6)}.png'),
                    normalize=True,
                    nrow=4,
                    value_range=(-0.5, 0.5),
                )
                save_image(
                    model.reverse(z_sample).cpu().data,
                    'flow_current.png',
                    normalize=True,
                    nrow=4,
                    value_range=(-0.5, 0.5),
                )
            model.train()

        if (i+1)%2500 == 0:
            torch.save(model.state_dict(), os.path.join(cpt_dir, f'iter{str(i+1).zfill(6)}_model.cpt'))
            torch.save(optimizer.state_dict(), os.path.join(cpt_dir, f'iter{str(i+1).zfill(6)}_optim.cpt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    # model
    parser.add_argument('--img_sz', type=int, default=64)
    parser.add_argument('--cond_sz', type=int) # CLEVR 24, CelebA 40
    parser.add_argument('--cond_fc_fts', type=int, default=64)
    parser.add_argument('--affine_conv_chs', type=int, default=256)
    parser.add_argument('--flow_depth', type=int, default=32)
    parser.add_argument('--num_levels', type=int, default=4)

    # training
    parser.add_argument('--num_iters', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--bs', type=int, default=16)

    parser.add_argument('--dataset', type=str, help='CLEVR, CelebA')
    parser.add_argument('--temp', default=0.8, type=float)
    parser.add_argument('--n_bits', default=5, type=int)

    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--cpt_dir', type=str, default='cpts')
    parser.add_argument('--result_dir', type=str, default='results')

    args = parser.parse_args()

    task_name = 'CondGLOW-{}-cond_sz{}-cond_fc_fts{}-affine_conv_chs{}-flow_depth{}-num_levels{}-img_sz{}-num_iters{}-lr{}-beta1{}-beta2{}-bs{}'.format(
        args.dataset, args.cond_sz, args.cond_fc_fts, args.affine_conv_chs,
        args.flow_depth, args.num_levels, args.img_sz, args.num_iters,
        args.lr, args.beta1, args.beta1, args.bs)
    log_dir = os.path.join(args.log_dir, task_name)
    cpt_dir = os.path.join(args.cpt_dir, task_name)
    result_dir = os.path.join(args.result_dir, task_name)

    model = CondGlow(3, args.flow_depth, args.num_levels, args.cond_sz, args.cond_fc_fts, args.affine_conv_chs)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(args.beta1, args.beta2))
    train(args, model, optimizer, log_dir, cpt_dir, result_dir)
