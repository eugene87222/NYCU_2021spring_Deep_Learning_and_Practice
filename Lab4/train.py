# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from radam import RAdam
from dataset import RetinopathyDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def downsample(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=stride, bias=False),
        nn.BatchNorm2d(out_ch))


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample_stride):
        super(BasicBlock, self).__init__()
        if downsample_stride is None:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.downsample = None
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.downsample = downsample(in_ch, out_ch, downsample_stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        ori = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            ori = self.downsample(ori)
        out = self.relu(out+ori)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, downsample_stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        if downsample_stride is None:
            self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.downsample = None
        else:
            self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=(3, 3), stride=downsample_stride, padding=(1, 1), bias=False)
            self.downsample = downsample(in_ch, out_ch, downsample_stride)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ori = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            ori = self.downsample(ori)
        out = self.relu(out+ori)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, None),
            BasicBlock(64, 64, None))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, (2, 2)),
            BasicBlock(128, 128, None))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, (2, 2)),
            BasicBlock(256, 256, None))
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, (2, 2)),
            BasicBlock(512, 512, None))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 256, (1, 1)),
            Bottleneck(256, 64, 256, None),
            Bottleneck(256, 64, 256, None))
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, 512, (2, 2)),
            Bottleneck(512, 128, 512, None),
            Bottleneck(512, 128, 512, None),
            Bottleneck(512, 128, 512, None))
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, 1024, (2, 2)),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None),
            Bottleneck(1024, 256, 1024, None))
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, (2, 2)),
            Bottleneck(2048, 512, 2048, None),
            Bottleneck(2048, 512, 2048, None))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 5)
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out


def train(model, train_loader, test_loader, optimizer, criterion, lr_decay, num_epoch, logger, cpt_dir):
    if lr_decay:
        scheduler = StepLR(optimizer, step_size=50, gamma=0.95)

    os.makedirs(cpt_dir, exist_ok=True)
    batch_done = 0
    epoch_pbar = tqdm(range(1, num_epoch+1))
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0
        batch_pbar = tqdm(train_loader)
        for batch_idx, (datas, labels) in enumerate(batch_pbar):
            inputs, targets = datas.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_done += 1
            epoch_loss += loss.cpu().item()
            logger.add_scalar('train-batch/loss', loss.item(), batch_done)

            batch_pbar.set_description(f'[train] [epoch:{epoch:>4}/{num_epoch}] [batch: {batch_idx+1:>5}/{len(train_loader)}] loss: {loss.item():.4f}')

        epoch_loss /= len(train_loader)
        torch.save(model.state_dict(), os.path.join(cpt_dir, f'epoch{epoch}.cpt'))
        if lr_decay:
            scheduler.step()

        _, train_acc, _, _ = evaluate(model, train_loader, criterion)
        test_avg_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)

        logger.add_scalar('train/loss', epoch_loss, epoch)
        logger.add_scalar('train/acc', train_acc, epoch)
        logger.add_scalar('test/loss', test_avg_loss, epoch)
        logger.add_scalar('test/acc', test_acc, epoch)

        epoch_pbar.set_description(f'[train] [epoch:{epoch:>4}/{num_epoch}] train acc: {train_acc:.4f}, test acc: {test_acc:.4f}')


def evaluate(model, loader, criterion=None):
    model.eval()
    correct = 0
    avg_loss = 0
    gt, pred = [], []
    with torch.no_grad():
        batch_pbar = tqdm(loader)
        for batch_idx, (datas, labels) in enumerate(batch_pbar):
            inputs, targets = datas.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                avg_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1)
            correct += (outputs==targets).sum().cpu().item()
            gt += targets.detach().cpu().numpy().tolist()
            pred += outputs.detach().cpu().numpy().tolist()

            if criterion is None:
                batch_pbar.set_description(f'[eval] [batch: {batch_idx+1:>5}/{len(loader)}] acc: {(outputs==targets).sum().item()/outputs.shape[0]:.4f}')
            else:
                batch_pbar.set_description(f'[eval] [batch: {batch_idx+1:>5}/{len(loader)}] acc: {(outputs==targets).sum().item()/outputs.shape[0]:.4f}, loss: {loss.item():.4f}')
        avg_loss /= len(loader)
        acc = correct / len(loader.dataset)

    if criterion is None:
        return None, acc, gt, pred
    else:
        return avg_loss, acc, gt, pred


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model',        type=str,   default='resnet18')
    parser.add_argument('--optimizer',    type=str,   default='sgd')
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum',     type=float, default=0.9)
    parser.add_argument('--num_epoch',    type=int,   default=10)
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--log_dir',      type=str,   default='logs')
    parser.add_argument('--cpt_dir',      type=str,   default='cpts')
    parser.add_argument('--pretrain',     action='store_true', default=False)
    parser.add_argument('--lr_decay',     action='store_true', default=False)
    args = parser.parse_args()

    if args.model == 'resnet18':
        if args.pretrain:
            model = models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 5)
        else:
            model = ResNet18()
    elif args.model == 'resnet50':
        if args.pretrain:
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 5)
        else:
            model = ResNet50()
    else:
        print('Unknown model')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    task_name = f'{args.model}_{args.optimizer}'
    if args.optimizer == 'sgd':
        task_name += f'_mmt{args.momentum}'
    task_name += f'_lr{args.lr}_wd{args.weight_decay}_{args.num_epoch}epoch_bs{args.batch_size}'
    if args.lr_decay:
        task_name += '_lr-decay'
    if args.pretrain:
        task_name += '_pretrain'
    print(task_name)
    logger = SummaryWriter(os.path.join(args.log_dir, task_name))

    train_dataset = RetinopathyDataset('../../diabetic_retinopathy_dataset', 'train')
    test_dataset = RetinopathyDataset('../../diabetic_retinopathy_dataset', 'test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    model.to(device)
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        lr_decay=args.lr_decay,
        num_epoch=args.num_epoch,
        logger=logger,
        cpt_dir=os.path.join(args.cpt_dir, task_name))
