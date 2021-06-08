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

from evaluator import evaluation_model
from CLEVR_dataset import CLEVRDataset
from gan import Generator, Discriminator, weights_init

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample_z(bs, n_z, mode='normal'):
    if mode == 'normal':
        return torch.normal(torch.zeros((bs, n_z)), torch.ones((bs, n_z)))
    elif mode == 'uniform':
        return torch.randn(bs, n_z)
    else:
        raise NotImplementedError()


def train(
        g_model, d_model, optimizer_g, optimizer_d, criterion,
        num_epochs, train_loader, test_loader, n_z, g_step,
        log_dir, cpt_dir, result_dir):
    logger = SummaryWriter(log_dir)
    os.makedirs(cpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    eval_model = evaluation_model()
    best_acc = 0
    batch_done = 0
    pbar_epoch = tqdm(range(num_epochs))
    for epoch in pbar_epoch:
        g_model.train()
        d_model.train()
        losses_g = 0
        losses_d = 0
        pbar_batch = tqdm(train_loader)
        for batch_idx, (images, conds) in enumerate(pbar_batch):
            images = images.to(device)
            conds = conds.to(device)

            bs = images.shape[0]
            real_label = torch.ones(bs).to(device)
            fake_label = torch.zeros(bs).to(device)

            # train discriminator
            optimizer_d.zero_grad()

            outputs = d_model(images, conds)
            loss_real = criterion(outputs, real_label)
            d_x = outputs.mean().item()

            z = sample_z(bs, n_z).to(device)
            fake_images = g_model(z, conds)
            outputs = d_model(fake_images.detach(), conds)
            loss_fake = criterion(outputs, fake_label)
            d_g_z1 = outputs.mean().item()

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # train generator
            optimizer_g.zero_grad()

            loss_g = 0
            d_g_z2 = 0
            for i in range(g_step):
                z = sample_z(bs, n_z).to(device)
                fake_images = g_model(z, conds)
                outputs = d_model(fake_images, conds)
                loss_g_part = criterion(outputs, real_label)
                d_g_z2_part = outputs.mean().item()

                loss_g_part.backward()
                optimizer_g.step()

                loss_g += loss_g_part
                d_g_z2 += d_g_z2_part
            loss_g /= g_step
            d_g_z2 /= g_step

            # outputs = d_model(fake_images, conds)
            # loss_g = criterion(outputs, real_label)
            # d_g_z2 = outputs.mean().item()

            # loss_g.backward()
            # optimizer_g.step()

            pbar_batch.set_description('[{}/{}][{}/{}][LossG={:.4f}][LossD={:.4f}][D(x)={:.4f}][D(G(z))={:.4f}/{:.4f}]'
                .format(epoch+1, num_epochs, batch_idx+1, len(train_loader), loss_g.item(), loss_d.item(), d_x, d_g_z1, d_g_z2))

            losses_g += loss_g.item()
            losses_d += loss_d.item()

            batch_done += 1
            logger.add_scalar('batch/loss_g', loss_g.item(), batch_done)
            logger.add_scalar('batch/loss_d', loss_d.item(), batch_done)
            logger.add_scalar('batch/d_x', d_x, batch_done)
            logger.add_scalar('batch/d_g_z1', d_g_z1, batch_done)
            logger.add_scalar('batch/d_g_z2', d_g_z2, batch_done)

        g_model.eval()
        d_model.eval()
        avg_acc = 0
        gen_images = None
        with torch.no_grad():
            for _, conds in enumerate(test_loader):
                conds = conds.to(device)
                z = sample_z(len(conds), n_z).to(device)
                fake_images = g_model(z, conds)
                fake_images = 0.5*fake_images + 0.5
                fake_images = torch.clamp(fake_images, min=0, max=1)
                print(
                    f'range=({fake_images.min()}, {fake_images.max()})',
                    file=open('gan_value_range.txt', 'w'))
                if gen_images is None:
                    gen_images = fake_images
                else:
                    gen_images = torch.vstack((gen_images, fake_images))
                acc = eval_model.eval(fake_images, conds)
                avg_acc += acc
        avg_acc /= len(test_loader)
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(
                g_model.state_dict(),
                os.path.join(cpt_dir, f'epoch{epoch+1}_eval-acc{avg_acc:.4f}.cpt'))

        pbar_epoch.set_description('[{}/{}][AvgLossG={:.4f}][AvgLossD={:.4f}][EvalAcc={:.4f}]'
            .format(epoch+1, num_epochs, losses_g/len(train_loader), losses_d/len(train_loader), avg_acc))
        logger.add_scalar('epoch/loss_g', losses_g/len(train_loader), epoch)
        logger.add_scalar('epoch/loss_d', losses_d/len(train_loader), epoch)
        logger.add_scalar('epoch/eval_acc', avg_acc, epoch)
        save_image(gen_images, os.path.join(result_dir, f'epoch{epoch+1}.png'), nrow=8)
        save_image(gen_images, 'gan_current.png', nrow=8)


if __name__ == '__main__':
    parser = ArgumentParser()
    # model
    parser.add_argument('--n_z', type=int, default=100)
    parser.add_argument('--num_conditions', type=int)  # CLEVR 24
    parser.add_argument('--n_c', type=int, default=100)
    parser.add_argument('--n_ch_g', type=int, default=64)
    parser.add_argument('--n_ch_d', type=int, default=64)
    parser.add_argument('--img_h', type=int, default=64)
    parser.add_argument('--img_w', type=int, default=64)
    parser.add_argument('--add_bias', action='store_true', default=False)

    # training
    parser.add_argument('--g_step', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--bs', type=int, default=128)

    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--cpt_dir', type=str, default='cpts')
    parser.add_argument('--result_dir', type=str, default='results')

    args = parser.parse_args()

    task_name = 'CondDCGAN-z{}-c{}-n_ch_g{}-n_ch_d{}-img_h{}-img_w{}-g_step{}-{}epoch-lr{}-beta1{}-beta2{}-bs{}'.format(
        args.n_z, args.n_c, args.n_ch_g, args.n_ch_d, args.img_h, args.img_w, args.g_step,
        args.num_epochs, args.lr, args.beta1, args.beta1, args.bs)
    if args.add_bias:
        task_name += '-add_bias'
    log_dir = os.path.join(args.log_dir, task_name)
    cpt_dir = os.path.join(args.cpt_dir, task_name)
    result_dir = os.path.join(args.result_dir, task_name)

    n_ch_g = [args.n_ch_g*8, args.n_ch_g*4, args.n_ch_g*2, args.n_ch_g]
    generator = Generator(args).to(device)
    generator.apply(weights_init)

    n_ch_d = [args.n_ch_d, args.n_ch_d*2, args.n_ch_d*4, args.n_ch_d*8]
    discriminator = Discriminator(args).to(device)
    discriminator.apply(weights_init)

    train_trans = transforms.Compose([
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CLEVRDataset('../../DLP_Lab7_dataset/task_1', train_trans, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    test_dataset = CLEVRDataset('../../DLP_Lab7_dataset/task_1', None, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = torch.optim.SGD(discriminator.parameters(), args.lr)

    train(
        g_model=generator,
        d_model=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        criterion=criterion,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        n_z=args.n_z,
        g_step=args.g_step,
        log_dir=log_dir,
        cpt_dir=cpt_dir,
        result_dir=result_dir)