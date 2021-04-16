# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from radam import RAdam
from dataloader import read_bci_data


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index,...]
        label = self.label[index]
        train_data = torch.tensor(data, dtype=torch.float32)
        train_label = torch.tensor(label, dtype=torch.float32)
        return data, label

    def __len__(self):
        return self.data.shape[0]


class EEGNet(nn.Module):
    def __init__(self, activation):
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
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True))

        self.depthwiseconv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25))

        self.separableconv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25))

        self.classify = nn.Sequential(nn.Linear(in_features=736, out_features=2, bias=True))


class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
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

        self.conv1 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(100),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5))

        self.conv1 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(200),
            self.activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5))

        self.classify = nn.Sequential(nn.Linear(in_features=43, out_features=2, bias=True))


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = read_bci_data()
    train_set = BCIDataset(train_data, train_label)
    test_set = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True)

    model = EEGNet('relu')
    # model = DeepConvNet('relu')
    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters(), lr=1e-3)
    num_epoch = 150
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0
        for batch_idx, (datas, labels) in enumerate(train_loader):
            inputs = datas.to(device)
            targets = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item().cpu()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            correct = 0
            for batch_idx, (datas, labels) in enumerate(test_loader):
                inputs = datas.to(device)
                targets = labels.to(device)

                outputs = model(inputs)
                outputs = torch.argmax(outputs, dim=1)
                correct += (outputs==targets).sum().item().cpu()
            acc = correct / len(test_loader.dataset)

        if epoch%10 == 0:
            print(epoch, epoch_loss)
            print(epoch, acc)
