# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan import Generator
from main_gan import evaluate
from evaluator import evaluation_model
from CLEVR_dataset import CLEVRDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    parser = ArgumentParser()
    # model
    parser.add_argument('--n_z', type=int, default=100)
    parser.add_argument('--num_conditions', type=int)  # CLEVR 24
    parser.add_argument('--n_c', type=int, default=100)
    parser.add_argument('--n_ch_g', type=int, default=64)
    parser.add_argument('--n_ch_d', type=int, default=64)
    parser.add_argument('--img_sz', type=int, default=64)
    parser.add_argument('--add_bias', action='store_true', default=False)

    parser.add_argument('--bs', type=int, default=32)

    parser.add_argument('--cpt_path', type=str, default='cpts')
    parser.add_argument('--output_dir', type=str, default='output')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fn = 'new_test'
    dataset = CLEVRDataset('../../DLP_Lab7_dataset/task_1', None, mode='test', test_json=f'{fn}.json')
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=8)

    generator = Generator(args).to(device)
    generator.load_state_dict(torch.load(args.cpt_path))
    eval_model = evaluation_model()

    avg_acc = 0
    for i in range(10):
        eval_acc, gen_images = evaluate(generator, loader, eval_model, args.n_z)
        avg_acc += eval_acc
        gen_images = 0.5*gen_images + 0.5
        save_image(gen_images, os.path.join(args.output_dir, f'conditional_DCGAN_{fn}.png'), nrow=8)
        print(f'acc: {eval_acc}')
    print(f'avg. acc: {avg_acc/10}')
