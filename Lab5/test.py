# -*- coding: utf-8 -*-
import os
import json
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from model import EncoderRNN, DecoderRNN
from dataset import OneHotEncoder, WordDataset
from train import load_model, generate_words, encode_decode, evaluation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_size', type=int, default=28)
    parser.add_argument('--hidden_input_size', type=int, default=256)
    parser.add_argument('--cond_size', type=int, default=4)
    parser.add_argument('--hidden_cond_size', type=int, default=8)
    parser.add_argument('--latent_size', type=int, default=32)

    parser.add_argument('--cpt_dir', type=str, required=True)
    parser.add_argument('--cpt_num', type=int, required=True)

    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))

    input_size = args.input_size
    hidden_input_size = args.hidden_input_size
    cond_size = args.cond_size
    hidden_cond_size = args.hidden_cond_size
    latent_size = args.latent_size

    cpt_dir = args.cpt_dir
    cpt_num = args.cpt_num
    encoder_cpt_path = os.path.join(cpt_dir, f'{cpt_num}_encoder.cpt')
    decoder_cpt_path = os.path.join(cpt_dir, f'{cpt_num}_decoder.cpt')

    train_dataset = WordDataset('.', 'train')
    test_dataset = WordDataset('.', 'test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    encoder = EncoderRNN(input_size, hidden_input_size, cond_size, hidden_cond_size, latent_size).to(device)
    decoder = DecoderRNN(input_size, hidden_input_size, hidden_cond_size, latent_size).to(device)
    tokenizer = OneHotEncoder()

    load_model(encoder, encoder_cpt_path, decoder, decoder_cpt_path)
    b, g, conv_res, gen_words = evaluation(encoder, decoder, tokenizer, test_loader, latent_size)

    for res in conv_res:
        print(f'input: {res[0]}')
        print(f'target: {res[1]}')
        print(f'prediction: {res[2]}\n')
    print(f'Average BLEU-4 score: {b:.4f}\n')

    for words in gen_words:
        print(words)
    print(f'Gaussian score: {g:.4f}')
