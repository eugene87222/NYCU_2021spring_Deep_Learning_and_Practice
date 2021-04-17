# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from dataloader import read_bci_data
from train import BCIDataset, EEGNet, DeepConvNet, evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model',      type=str, default='EEGNet')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cpt_path',   type=str, required=True)
    args = parser.parse_args()

    model = vars()[args.model](args.activation)
    model.load_state_dict(torch.load(args.cpt_path))

    train_data, train_label, test_data, test_label = read_bci_data()
    train_set = BCIDataset(train_data, train_label)
    test_set = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model.to(device)
    _, acc = evaluate(model, test_loader)
    print(f'acc: {100*acc:.2f}%')
