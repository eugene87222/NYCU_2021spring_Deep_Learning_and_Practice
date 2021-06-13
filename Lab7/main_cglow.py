# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from cglow import CondGlowModel
from evaluator import evaluation_model
from CLEVR_dataset import CLEVRDataset
from CelebA_dataset import CelebADataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(model, loader, images, cur_z, cur_conds, num_samples, eval_model):
    model.eval()
    if loader is None:
        gen_images = []
        for _ in range(images.shape[0]):
            gen_images.append([])
        with torch.no_grad():
            for _ in range(num_samples):
                fake_images, _ = model(x=None, cond=cur_conds, reverse=True)
                for i in range(fake_images.shape[0]):
                    gen_images[i].append(fake_images[i])
            images_inv, _ = model(x=cur_z, cond=cur_conds, reverse=True)
        output = None
        for b in range(images.shape[0]):
            row = torch.cat((images[b], images_inv[b]), dim=2)
            for i in range(len(gen_images[b])):
                row = torch.cat((row, gen_images[b][i]), dim=2)
            if output is None:
                output = row
            else:
                output = torch.cat((output, row), dim=1)
        result = [output]
    else:
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
        result = [avg_acc, gen_images]
    return result


def train(
        model, optimizer, scheduler,
        num_epochs, train_loader, test_loader,
        num_samples, eval_interval, log_dir, cpt_dir, result_dir):
    logger = SummaryWriter(log_dir)
    os.makedirs(cpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    if test_loader is not None:
        eval_model = evaluation_model()
    else:
        eval_model = None
    best_acc = 0
    batch_done = 0
    pbar_epoch = tqdm(range(num_epochs))

    for epoch in pbar_epoch:
        model.train()
        mean_nll = 0
        pbar_batch = tqdm(train_loader)
        for batch_idx, (images, conds) in enumerate(pbar_batch):
            images = images.to(device)
            conds = conds.to(device)

            optimizer.zero_grad()
            z, nll = model(images, conds)
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
                result = evaluate(model, test_loader, images, z, conds, num_samples, eval_model)
                # CLEVR
                if len(result) == 2:
                    eval_acc, gen_images = result
                    print(
                        f'range=({gen_images.min()}, {gen_images.max()})',
                        file=open('flow_value_range.txt', 'w'))
                    gen_images = 0.5*gen_images + 0.5
                    logger.add_scalar('batch/eval_acc', eval_acc, batch_done)
                    if eval_acc > best_acc:
                        best_acc = eval_acc
                        state = {
                            'model': model.state_dict(),
                            'optim': optimizer.state_dict()
                        }
                        torch.save(
                            state,
                            os.path.join(cpt_dir, f'epoch{epoch+1}_iter{batch_done}_eval-acc{eval_acc:.4f}.cpt'))
                # CelebA
                else:
                    gen_images = result[0]
                    print(
                        f'range=({gen_images.min()}, {gen_images.max()})',
                        file=open('flow_value_range.txt', 'w'))
                save_image(gen_images, os.path.join(result_dir, f'epoch{epoch+1}_iter{batch_done}.png'), nrow=8)
                save_image(gen_images, 'flow_current.png', nrow=8)
            model.train()

        if scheduler is not None:
            scheduler.step()

        mean_nll /= len(train_loader)
        if test_loader is not None:
            eval_acc, gen_images = evaluate(model, test_loader, None, None, None, None, eval_model)
            print(
                f'range=({gen_images.min()}, {gen_images.max()})',
                file=open('flow_value_range.txt', 'w'))
            gen_images = 0.5*gen_images + 0.5
            if eval_acc > best_acc:
                best_acc = eval_acc
                state = {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict()
                }
                torch.save(
                    state,
                    os.path.join(cpt_dir, f'epoch{epoch+1}_last_eval-acc{eval_acc:.4f}.cpt'))
            save_image(gen_images, os.path.join(result_dir, f'epoch{epoch+1}_last.png'), nrow=8)
            save_image(gen_images, 'flow_current.png', nrow=8)
            pbar_epoch.set_description('[Epoch {}/{}][Mean NLL={:.4f}][Eval Acc={:.4f}]'
                .format(epoch+1, num_epochs, mean_nll, eval_acc))
            logger.add_scalar('epoch/eval_acc', eval_acc, epoch+1)
        else:
            pbar_epoch.set_description('[Epoch {}/{}][Mean NLL={:.4f}]'
                .format(epoch+1, num_epochs, mean_nll))
        logger.add_scalar('epoch/mean_nll', mean_nll, epoch+1)


if __name__ == '__main__':
    parser = ArgumentParser()
    # model
    parser.add_argument('--input_sz', type=int, nargs='+', default=[3, 64, 64])
    parser.add_argument('--cond_sz', type=int) # CLEVR 24, CelebA 40
    parser.add_argument('--cond_fc_fts', type=int, default=64)
    parser.add_argument('--affine_conv_chs', type=int, default=256)
    parser.add_argument('--flow_depth', type=int, default=8)
    parser.add_argument('--num_levels', type=int, default=5)
    parser.add_argument('--learn_prior', action='store_true', default=False)
    parser.add_argument('--img_h', type=int, default=64)
    parser.add_argument('--img_w', type=int, default=64)

    # training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--bs', type=int, default=20)

    parser.add_argument('--dataset', type=str, help='CLEVR, CelebA')

    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--cpt_dir', type=str, default='cpts')
    parser.add_argument('--result_dir', type=str, default='results')

    args = parser.parse_args()

    task_name = 'CondGLOW-{}-cond_sz{}-cond_fc_fts{}-affine_conv_chs{}-flow_depth{}-num_levels{}-img_h{}-img_w{}-{}epoch-lr{}-beta1{}-beta2{}-bs{}'.format(
        args.dataset, args.cond_sz, args.cond_fc_fts, args.affine_conv_chs,
        args.flow_depth, args.num_levels, args.img_h, args.img_w, args.num_epochs,
        args.lr, args.beta1, args.beta1, args.bs)
    log_dir = os.path.join(args.log_dir, task_name)
    cpt_dir = os.path.join(args.cpt_dir, task_name)
    result_dir = os.path.join(args.result_dir, task_name)

    if args.dataset == 'CLEVR':
        train_trans = transforms.Compose([
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = CLEVRDataset('../../DLP_Lab7_dataset/task_1', train_trans, mode='train')
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=8)
        test_dataset = CLEVRDataset('../../DLP_Lab7_dataset/task_1', None, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=8)
    elif args.dataset == 'CelebA':
        trans = transforms.Compose([
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor()
        ])
        dataset = CelebADataset('../../DLP_Lab7_dataset/task_2', trans)
        train_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=8)
        test_loader = None
    else:
        raise NotImplementedError()

    model = CondGlowModel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(args.beta1, args.beta2))
    lr_lambda = lambda epoch: min(1.0, (epoch+1)/5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        num_samples=args.num_samples,
        eval_interval=args.eval_interval,
        log_dir=log_dir,
        cpt_dir=cpt_dir,
        result_dir=result_dir)
