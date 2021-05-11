# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_log(log_dir, log_fn):
    log = json.load(open(os.path.join(log_dir, log_fn)))
    return np.array(log)


if __name__ == '__main__':
    for folder in os.listdir('logs_json'):
        log_dir = os.path.join('logs_json', folder)

        ce_fn = 'run-train_CrossEntropy-tag-train.json'
        kld_fn = 'run-train_KLD-tag-train.json'

        tfr_fn = 'run-train_Teacher ratio-tag-train.json'
        kldw_fn = 'run-train_KLD weight-tag-train.json'
        bleu_fn = 'run-train_BLEU4 score-tag-train.json'
        g_fn = 'run-train_Gaussian score-tag-train.json'

        ce = load_log(log_dir, ce_fn)
        kld = load_log(log_dir, kld_fn)

        tfr = load_log(log_dir, tfr_fn)
        kldw = load_log(log_dir, kldw_fn)
        bleu = load_log(log_dir, bleu_fn)
        g = load_log(log_dir, g_fn)

        fs = 12

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8), dpi=300)

        axs[0].plot(kld[:,1], kld[:,2], c='tab:blue', lw=2, label='KLD')
        axs[0].set_ylabel('KLD', fontsize=fs)

        axs[1].plot(ce[:,1], ce[:,2], c='tab:orange', lw=2, label='Cross Entropy')
        axs[1].set_ylabel('CE', fontsize=fs)

        axs[2].scatter(bleu[:,1], bleu[:,2], c='tab:green', s=2, label='BLEU-4 score')
        axs[2].scatter(g[:,1], g[:,2], s=2, c='tab:brown', label='Gaussian score')
        axs[2].set_ylabel('Score', fontsize=fs)

        axs[3].plot(tfr[:,1], tfr[:,2], c='tab:purple', label='Teacher ratio')
        axs[3].plot(kldw[:,1], kldw[:,2], c='tab:red', label='KLD weight')
        axs[3].set_ylabel('Ratio/Weight', fontsize=fs)
        axs[3].set_xlabel('Per 500 iterations', fontsize=fs)

        for i, ax in enumerate(axs):
            ax.yaxis.set_label_coords(-0.07, 0.5)
            ax.grid()

        fig.legend()
        fig.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.12)
        fn = log_dir.split('/')[-1] + '_curve.png'
        plt.savefig(os.path.join('plots', fn), dpi=300)

        fig, axs = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

        bleu_score = np.zeros(bleu.shape[0])
        bleu_score[bleu[:,2]>=0.4] = 80
        bleu_score[bleu[:,2]>=0.6] = 90
        bleu_score[bleu[:,2]>=0.7] = 100

        g_score = np.zeros(g.shape[0])
        g_score[g[:,2]>=0.05] = 80
        g_score[g[:,2]>=0.2] = 90
        g_score[g[:,2]>=0.3] = 100

        total_score = bleu_score*0.2 + g_score*0.1

        max_score = np.max(total_score)
        max_idx = (total_score==max_score).astype(np.int32)
        max_idx = bleu[:,1][(bleu[:,1]*max_idx)>0]
        print(log_dir.split('/')[-1])
        print(max_idx)

        axs.set_yticks(range(31))
        axs.grid()
        axs.plot(bleu[:,1], bleu_score*0.2+g_score*0.1, label='total')
        axs.plot(bleu[:,1], bleu_score*0.2, label='BLEU')
        axs.plot(g[:,1], g_score*0.1, label='Gaussian')
        fn = log_dir.split('/')[-1] + '_score.png'
        fig.legend()
        plt.savefig(os.path.join('plots', fn), dpi=300)
