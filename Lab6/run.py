# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--target_freq', type=int)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    task_name = f'{args.model}_{args.episode}episode_{args.capacity}capacity'
    if args.target_freq is not None:
        task_name += f'_{args.target_freq}target_freq'
    if args.tau is not None:
        task_name += f'_{args.tau}tau'

    if args.test_only:
        command = f'python {args.model}.py -m {task_name}.pth --episode {args.episode} --seed {args.seed} --capacity {args.capacity} --logdir log/{task_name}-test --test_only'
        if args.render:
            command += ' --render'
    else:
        command = f'python {args.model}.py -m {task_name}.pth --episode {args.episode} --seed {args.seed} --capacity {args.capacity} --logdir log/{task_name}'

    if args.target_freq is not None:
        command += f' --target_freq {args.target_freq}'
    if args.tau is not None:
        command += f' --tau {args.tau}'

    os.system(command)
