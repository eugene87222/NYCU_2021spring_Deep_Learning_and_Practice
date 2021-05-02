# -*- coding: utf-8 -*-
import re
import json
import numpy as np
from glob import glob
from matplotlib import rcParams
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

if __name__ == '__main__':
    rcParams['font.family'] = 'monospace'
    c = {
        '': ['limegreen'],
        'pretrain': ['royalblue']
    }
    ls = {
        'train': '-',
        'test': (0, (4, 1, 2, 1))
    }

    length = 20
    for model in ['resnet18', 'resnet50']:
        for bs in [4, 8, 16]:
            for thres in [75, 80, 82]:
                plt.plot([1, length], [thres, thres], color='red', lw=1)
                plt.gca().text(length+0.2, thres, f'{thres}%', color='red')

            logfiles = sorted(glob(f'val_acc/run-{model}_sgd_*_bs{bs}*'))
            for logfile in logfiles:
                data = json.load(open(logfile, 'r'))
                data = np.array(data)
                res = re.findall(rf'run-{model}_sgd_mmt(.+)_lr(.+)_wd(.+)_(.+)epoch_bs{bs}_?(pretrain)?-tag-(.+)_acc', logfile, re.S)[0]
                highest = data[:,2].max() * 100

                label = f'{res[-1]:<5}'
                if len(res[-2]):
                    label += ' (pretrain)'
                label += f' ({highest:.1f}%)'
                plt.plot(data[:,1], 100*data[:,2], linestyle=ls[res[-1]], label=label, lw=1.5, c=c[res[-2]][0])

            title = f'Accuracy comparison ({model})\nbatch size = {bs}'
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy(%)')
            plt.ylim(72, 88)
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{model}-compare_pretrain_bs{bs}.png', dpi=300)
            plt.clf()
