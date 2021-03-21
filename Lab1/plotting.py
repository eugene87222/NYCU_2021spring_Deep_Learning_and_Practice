# -*- coding: utf-8 -*-
import re
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_type = 'xor'
    lr = '1'
    num_features = '*'

    title = []
    if data_type != '*':
        title.append(f'Data type: {data_type}')
    if lr != '*':
        title.append(f'Learning rate: {lr}')
    if num_features != '*':
        title.append(f'Network: {num_features}')
    title = '\n'.join(title)
    plt.title(title)

    pattern = f'train_loss_exps/{data_type}_lr{lr}_{num_features}_lc.npy'
    fns = sorted(glob(pattern))
    for fn in fns:
        losses = np.load(fn)
        pattern = '' 
        pattern += data_type if data_type!='*' else '(.*)'
        pattern += f'_lr{lr}' if lr!='*' else '_lr(.*)'
        pattern += f'_{num_features}' if num_features!='*' else '_(.*)'
        pattern += '_lc'

        items = re.findall(rf'{pattern}', fn)[0]
        if type(items) is tuple:
            items = list(reversed(items))
        else:
            items = [items]
        label = ''
        label += f', Data type: {items.pop()}' if data_type=='*' else ''
        label += f', Learning rate: {items.pop()}' if lr=='*' else ''
        label += f', Network: {items.pop()}' if num_features=='*' else ''
        plt.plot(np.arange(len(losses)), losses, label=label.strip(',').strip(), lw=2)

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.show()
