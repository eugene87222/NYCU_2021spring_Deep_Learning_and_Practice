# -*- coding: utf-8 -*-
import re
import json
import numpy as np
from glob import glob
from matplotlib import rcParams
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from dataloader import read_bci_data

if __name__ == '__main__':
    rcParams['font.family'] = 'monospace'
    activ = {
        'elu': ['ELU', 'darkorange'],
        'relu': ['ReLU', 'royalblue'],
        'lrelu': ['Leaky ReLU', 'limegreen']
    }
    ls = {
        'train': '-',
        'test': (0, (4, 1, 2, 1))
    }

    plot_result = True
    plot_data = False

    if plot_result:
        for model in ['DeepConvNet', 'EEGNet']:
            plt.plot([1, 300], [87, 87], color='red', lw=1)
            plt.gca().text(301.5, 87, '87%', color='red')
            plt.plot([1, 300], [85, 85], color='red', lw=1)
            plt.gca().text(301.5, 85, '85%', color='red')

            logfiles = sorted(glob(f'compare_best/run-{model}*'))
            for logfile in logfiles:
                data = json.load(open(logfile, 'r'))
                data = np.array(data)
                res = re.findall(rf'run-{model}_(.+)_lr(.+)_wd(.+)_(.+)epoch_bs(.+)-tag-(.+)_acc', logfile, re.S)[0]
                highest = data[:,2].max() * 100
                label = f'{activ[res[0]][0]:<10} | wd={res[2]:<5} | bs={res[-2]} | {res[-1]:<5} ({highest:.1f}%)'
                plt.plot(data[:,1], 100*data[:,2], linestyle=ls[res[-1]], label=label, lw=1.5, c=activ[res[0]][1])

            title = f'Accuracy comparison ({model})'
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy(%)')
            # plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{model}-compare-best.png', dpi=300)
            plt.clf()


        for model in ['DeepConvNet', 'EEGNet']:
            plt.plot([1, 300], [87, 87], color='red', lw=1)
            plt.gca().text(301.5, 87, '87%', color='red')
            plt.plot([1, 300], [85, 85], color='red', lw=1)
            plt.gca().text(301.5, 85, '85%', color='red')

            logfiles = sorted(glob(f'compare_activation/run-{model}*'))
            for logfile in logfiles:
                data = json.load(open(logfile, 'r'))
                data = np.array(data)
                res = re.findall(rf'run-{model}_(.+)_lr(.+)_wd(.+)_(.+)epoch_bs(.+)-tag-(.+)_acc', logfile, re.S)[0]
                highest = data[:,2].max() * 100
                label = f'{activ[res[0]][0]:<10} | {res[-1]:<5} ({highest:.1f}%)'
                plt.plot(data[:,1], 100*data[:,2], linestyle=ls[res[-1]], label=label, lw=1.5, c=activ[res[0]][1])

            title = f'Activation function comparison ({model})\nweight decay = {res[2]}\nbatch size = {res[-2]}'
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy(%)')
            # plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{model}-compare-activation.png', dpi=300)
            plt.clf()

    if plot_data:
        train_data, train_label, test_data, test_label = read_bci_data()
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        axs[0].plot(np.arange(train_data.shape[-1]), train_data[0,0,0,...], c='royalblue', lw=1.5)
        axs[0].set_title('Channel 1')
        axs[1].plot(np.arange(train_data.shape[-1]), train_data[0,0,1,...], c='royalblue', lw=1.5)
        axs[1].set_title('Channel 2')
        plt.savefig('data.png', dpi=300)
