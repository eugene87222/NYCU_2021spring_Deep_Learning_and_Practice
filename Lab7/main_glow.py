# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from glow import GlowModel
from CelebA_dataset import CelebADataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(model, num_samples=16):
    model.eval()
    gen_images = []
    max_val, min_val = None, None
    with torch.no_grad():
        for _ in range(num_samples):
            fake_image, _ = model(x=None, bs=1, reverse=True)
            gen_images.append(fake_image[0])
            if min_val is None:
                min_val = fake_image.min()
                max_val = fake_image.max()
            else:
                min_val = min(min_val, fake_image.min())
                max_val = min(max_val, fake_image.max())
    return gen_images, min_val, max_val


def train(
        model, optimizer,
        num_epochs, train_loader,
        num_samples, eval_interval, log_dir, cpt_dir, result_dir):
    logger = SummaryWriter(log_dir)
    os.makedirs(cpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    min_nll = 1e5
    best_acc = 0
    batch_done = 0
    pbar_epoch = tqdm(range(num_epochs))
    for epoch in pbar_epoch:
        model.train()
        mean_nll = 0
        pbar_batch = tqdm(train_loader)
        for batch_idx, images in enumerate(pbar_batch):
            images = images.to(device)

            optimizer.zero_grad()
            z, nll = model(images)
            loss = torch.mean(nll)
            mean_nll = mean_nll + loss.item()
            loss.backward()
            optimizer.step()

            batch_done += 1
            logger.add_scalar('batch/nll', loss.item(), batch_done)
            pbar_batch.set_description('[Epoch {}/{}][Batch {}/{}][{}][NLL={:.4f}]'
                .format(
                    epoch+1, num_epochs, batch_idx+1, len(train_loader),
                    batch_done%eval_interval, loss.item()))

            if batch_done%eval_interval == 0:
                gen_images, min_val, max_val = evaluate(model, num_samples=num_samples)
                gen_images = torch.Tensor(gen_images)
                print(f'range=({min_val}, {max_val})', file=open('flow_value_range.txt', 'w'))
                save_image(gen_images, os.path.join(result_dir, f'epoch{epoch+1}_iter{batch_done}.png'), nrow=8)
                save_image(gen_images, 'flow_current.png', nrow=8)
            model.train()

        mean_nll /= len(train_loader)
        if mean_nll < min_nll:
            min_nll = mean_nll
            state = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict()
            }
            torch.save(
                state,
                os.path.join(cpt_dir, f'epoch{epoch+1}_mean-nll{mean_nll:.4f}.cpt'))

        pbar_epoch.set_description('[Epoch {}/{}][Mean NLL={:.4f}]'
            .format(epoch+1, num_epochs, mean_nll))
        logger.add_scalar('epoch/mean_nll', mean_nll, epoch+1)


if __name__ == '__main__':
    parser = ArgumentParser()
    # model
    parser.add_argument('--input_sz', type=int, nargs='+', default=[3, 64, 64])
    parser.add_argument('--affine_conv_chs', type=int, default=256)
    parser.add_argument('--flow_depth', type=int, default=8)
    parser.add_argument('--num_levels', type=int, default=5)
    parser.add_argument('--learn_prior', action='store_true', default=False)
    parser.add_argument('--img_h', type=int, default=64)
    parser.add_argument('--img_w', type=int, default=64)

    # training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--bs', type=int, default=20)

    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--cpt_dir', type=str, default='cpts')
    parser.add_argument('--result_dir', type=str, default='results')

    args = parser.parse_args()

    task_name = 'GLOW-affine_conv_chs{}-flow_depth{}-num_levels{}-img_h{}-img_w{}-{}epoch-lr{}-beta1{}-beta2{}-bs{}'.format(
        args.affine_conv_chs, args.flow_depth, args.num_levels,
        args.img_h, args.img_w, args.num_epochs,
        args.lr, args.beta1, args.beta1, args.bs)
    log_dir = os.path.join(args.log_dir, task_name)
    cpt_dir = os.path.join(args.cpt_dir, task_name)
    result_dir = os.path.join(args.result_dir, task_name)

    trans = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor()
    ])
    dataset = CelebADataset('../../DLP_Lab7_dataset/task_2', trans, cond=False)
    train_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=8)

    model = GlowModel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(args.beta1, args.beta2))
    train(
        model=model,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        num_samples=args.num_samples,
        eval_interval=args.eval_interval,
        log_dir=log_dir,
        cpt_dir=cpt_dir,
        result_dir=result_dir)
