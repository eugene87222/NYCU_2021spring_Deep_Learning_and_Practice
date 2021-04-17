# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from radam import RAdam
from dataloader import read_bci_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]


class EEGNet(nn.Module):
    def __init__(self, activation='relu'):
        super(EEGNet, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        else:
            print('Unknown activation function')
            raise NotImplementedError

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16))

        self.depthwiseconv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25))

        self.separableconv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25))

        self.classify = nn.Sequential(nn.Linear(in_features=736, out_features=2, bias=True))

    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseconv(out)
        out = self.separableconv(out)
        out = self.classify(out.flatten(start_dim=1))
        return out


class DeepConvNet(nn.Module):
    def __init__(self, activation='relu'):
        super(DeepConvNet, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            print('Unknown activation function')
            raise NotImplementedError

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(25),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5))

        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(50),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5))

        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(100),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5))

        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(200),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5))

        self.classify = nn.Sequential(nn.Linear(in_features=43*200, out_features=2, bias=True))

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.classify(out.flatten(start_dim=1))
        return out


def train(model, train_loader, test_loader, optimizer, criterion, lr_decay, num_epoch, logger, cpt_dir):
    if lr_decay:
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    os.makedirs(cpt_dir, exist_ok=True)
    batch_done = 0
    epoch_pbar = tqdm(range(1, num_epoch+1))
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0
        for batch_idx, (datas, labels) in enumerate(train_loader):
            inputs, targets = datas.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_done += 1
            epoch_loss += loss.cpu().item()
            logger.add_scalar('train-batch/loss', loss.item(), batch_done)

        epoch_loss /= len(train_loader)
        torch.save(model.state_dict(), os.path.join(cpt_dir, f'epoch{epoch}.cpt'))
        if lr_decay:
            scheduler.step()

        _, train_acc = evaluate(model, train_loader, criterion)
        test_avg_loss, test_acc = evaluate(model, test_loader, criterion)

        logger.add_scalar('train/loss', epoch_loss, epoch)
        logger.add_scalar('train/acc', train_acc, epoch)
        logger.add_scalar('test/loss', test_avg_loss, epoch)
        logger.add_scalar('test/acc', test_acc, epoch)

        epoch_pbar.set_description(f'[epoch:{epoch:>4}/{num_epoch}] train acc: {train_acc:.4f}, test acc: {test_acc:.4f}')


def evaluate(model, loader, criterion=None):
    model.eval()
    correct = 0
    avg_loss = 0
    with torch.no_grad():
        for batch_idx, (datas, labels) in enumerate(loader):
            inputs, targets = datas.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)
                avg_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1)
            correct += (outputs==targets).sum().cpu().item()
        avg_loss /= len(loader)
        acc = correct / len(loader.dataset)

    if criterion is None:
        return None, acc
    else:
        return avg_loss, acc


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model',        type=str,   default='EEGNet')
    parser.add_argument('--activation',   type=str,   default='relu')
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch',    type=int,   default=300)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--log_dir',      type=str,   default='logs')
    parser.add_argument('--cpt_dir',      type=str,   default='cpts')
    parser.add_argument('--lr_decay',     action='store_true', default=False)
    args = parser.parse_args()

    model = vars()[args.model](args.activation)
    lr = args.lr
    weight_decay = args.weight_decay
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    lr_decay = args.lr_decay
    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    task_name = f'{args.model}_{args.activation}_lr{lr}_wd{weight_decay}_{num_epoch}epoch_bs{batch_size}'
    if lr_decay:
        task_name += '_lr-decay'
    print(task_name)
    logger = SummaryWriter(os.path.join(args.log_dir, task_name))

    train_data, train_label, test_data, test_label = read_bci_data()
    train_set = BCIDataset(train_data, train_label)
    test_set = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model.to(device)
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        lr_decay=lr_decay,
        num_epoch=num_epoch,
        logger=logger,
        cpt_dir=os.path.join(args.cpt_dir, task_name))
