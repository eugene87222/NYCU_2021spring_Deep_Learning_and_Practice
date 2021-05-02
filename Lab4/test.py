# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from dataset import RetinopathyDataset
from train import ResNet18, ResNet50, evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_confusion_matrix(y_true, y_pred, labels, fn):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)), normalize='true')
    fig, ax = plt.subplots()
    sn.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='.1f')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground truth')
    ax.xaxis.set_ticklabels(labels, rotation=45)
    ax.yaxis.set_ticklabels(labels, rotation=0)
    plt.title('Normalized comfusion matrix')
    plt.savefig(fn, dpi=300)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model',      type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cpt_path',   type=str, required=True)
    args = parser.parse_args()

    if args.model == 'resnet18':
        model = ResNet18()
    elif args.model == 'resnet50':
        model = ResNet50()
    else:
        raise NotImplementedError
    model.load_state_dict(torch.load(args.cpt_path))
    task_name = args.cpt_path.split('/')[-2]
    res = re.findall(rf'(.+)_sgd_mmt(.+)_lr(.+)_wd(.+)_(.+)epoch_bs(\d+)_?(pretrain)?', task_name, re.S)[0]
    fn = f'{res[0]}_{res[5]}'
    if len(res[6]):
        fn += f'_{res[6]}'

    test_dataset = RetinopathyDataset('../../diabetic_retinopathy_dataset', 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

    model.to(device)
    _, acc, gt, pred = evaluate(model, test_loader)
    plot_confusion_matrix(gt, pred, [0, 1, 2, 3, 4], os.path.join('cm', f'{fn}.png'))
    print(task_name)
    print(f'acc: {100*acc:.2f}%')
