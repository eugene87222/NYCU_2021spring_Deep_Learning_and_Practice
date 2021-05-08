# -*- coding: utf-8 -*-
import os
import json
import random
import string
import numpy as np
from io import open
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# plt.switch_backend('agg')

MAX_LEN = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OneHotEncoder():
    def __init__(self):
        self.char2token = {}
        self.token2char = {}
        for i in range(26):
            self.char2token[chr(ord('a')+i)] = i
            self.token2char[i] = chr(ord('a')+i)
        self.char2token['SOS'] = 26
        self.token2char[26] = 'SOS'
        self.char2token['EOS'] = 27
        self.token2char[27] = 'EOS'
        self.num_token = 26 + 2

    def tokenize(self, word):
        chars = ['SOS'] + list(word) + ['EOS']
        return [self.char2token[char] for char in chars]

    def inv_tokenize(self, vec, show_token=False, check_end=True):
        word = ''
        for v in vec:
            char = self.token2char[v.item()]
            if len(char) > 1:
                ch = f'<{char}>' if show_token else ''
            else:
                ch = char
            word += ch
            if check_end and char == 'EOS':
                break
        return word


class WordDataset(Dataset):
    def __init__(self, txt_dir, mode='train'):
        '''All tenses
        ---
        simple present(sp), third person(tp), present progressive(pg), simple past(p)

        Tenses in test.txt
        ---
        sp -> p  
        sp -> pg  
        sp -> tp  
        sp -> tp  
        p  -> tp  
        sp -> pg  
        p  -> sp  
        pg -> sp  
        pg -> p  
        pg -> tp'''

        txt_path = os.path.join(txt_dir, f'{mode}.txt')
        self.data = np.loadtxt(txt_path, dtype=str)
        self.tense = ['sp', 'tp', 'pg', 'p']
        self.mode = mode

        if mode == 'train':
            self.data = self.data.reshape(-1)
        
        if mode == 'test':
            self.conversion = np.array([
                [0, 3],
                [0, 2],
                [0, 1],
                [0, 1],
                [3, 1],
                [0, 2],
                [3, 0],
                [2, 0],
                [2, 3],
                [2, 1]
            ])

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.data[idx], idx%len(self.tense) 
        else:
            return self.data[idx,0], self.conversion[idx,0], self.data[idx,1], self.conversion[idx, 1]

    def __len__(self):
        return self.data.shape[0]


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_input_size, cond_size, hidden_cond_size, latent_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_input_size = hidden_input_size
        self.cond_size = cond_size
        self.hidden_cond_size = hidden_cond_size
        self.latent_size = latent_size

        self.embed_word = nn.Embedding(input_size, hidden_input_size)
        self.embed_cond = nn.Embedding(cond_size, hidden_cond_size)
        self.lstm = nn.LSTM(hidden_input_size, hidden_input_size)
        self.mean = nn.Linear(hidden_input_size, latent_size)
        self.logvar = nn.Linear(hidden_input_size, latent_size)

    def forward(self, inputs, hidden, cell, cond):
        cond_embedded = self.embed_cond(cond).reshape(1, 1, -1)
        hidden = torch.cat((hidden, cond_embedded), dim=2)

        inputs_embedded = self.embed_word(inputs).reshape(-1, 1, self.hidden_input_size)
        outputs, (hidden, cell) = self.lstm(inputs_embedded, (hidden, cell))

        m = self.mean(hidden)
        lv = self.logvar(hidden)
        epsilon = torch.normal(torch.zeros(self.latent_size), torch.ones(self.latent_size)).to(device)
        z = torch.exp(lv/2)*epsilon + m
        return m, lv, z

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_input_size-self.hidden_cond_size, device=device)

    def init_cell(self):
        return torch.zeros(1, 1, self.hidden_input_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_input_size, hidden_cond_size, latent_size):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_input_size = hidden_input_size
        self.hidden_cond_size = hidden_cond_size
        self.latent_size = latent_size

        self.latent_to_hidden = nn.Linear(latent_size+hidden_cond_size, hidden_input_size)
        self.embed_word = nn.Embedding(input_size, hidden_input_size)
        self.lstm = nn.LSTM(hidden_input_size, hidden_input_size)
        self.out = nn.Linear(hidden_input_size, input_size)

    def forward(self, inputs, hidden, cell):
        inputs_embedded = self.embed_word(inputs).reshape(-1, 1, self.hidden_input_size)
        inputs_embedded = F.relu(inputs_embedded)
        output, (hidden, cell) = self.lstm(inputs_embedded, (hidden, cell))
        output = self.out(output).reshape(-1, self.input_size)
        return output, hidden, cell

    def init_hidden(self, z, cond_embedded):
        latent = torch.cat((z, cond_embedded), dim=2)
        return self.latent_to_hidden(latent)

    def init_cell(self):
        return torch.zeros(1, 1, self.hidden_input_size, device=device)


# compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


def gaussian_score(words):
    words_list = []
    score = 0
    yourpath = 'train.txt'
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)


def generate_words(encoder, decoder, latent_size, tokenizer):
    decoder.eval()
    sos_token = tokenizer.char2token['SOS']
    eos_token = tokenizer.char2token['EOS']
    words = []
    with torch.no_grad():
        for i in range(100):
            z = torch.normal(torch.zeros(1, 1, latent_size), torch.ones(1, 1, latent_size)).to(device)
            words.append([])
            for cond in range(4):
                cond = to_long([cond]).to(device)
                cond_embedded = encoder.embed_cond(cond).reshape(1, 1, -1).to(device)
                output = inference(decoder, z, cond_embedded, None, None, sos_token, eos_token, MAX_LEN)
                output_token = torch.max(torch.softmax(output, dim=1), 1)[1]
                output_word = tokenizer.inv_tokenize(output_token)
                words[-1].append(output_word)
    return words


def kld_weight_annealing(epoch, num_epoch, final_weight, cycle_num):
    num_epoch //= cycle_num
    epoch %= num_epoch
    thres = int(num_epoch*0.2*cycle_num)
    if epoch < thres:
        w = 0
    else:
        w = (epoch-thres) * final_weight / (num_epoch-thres)
    w = max(0, w)
    w = min(1, w)
    return w


def teacher_forcing_ratio(epoch, num_epoch, final_ratio):
    thres = int(num_epoch*0.2)
    if epoch < thres:
        w = 1
    else:
        w = final_ratio + (num_epoch-epoch) * (1-final_ratio) / (num_epoch-thres)
    w = max(0, w)
    w = min(1, w)
    return w


def kld_loss(m, lv):
    return 0.5 * torch.sum(torch.exp(lv)+m**2-1-lv)


def inference(decoder, z, cond_embedded, teacher_token, tf_r, sos_token, eos_token, max_len):
    hidden = decoder.init_hidden(z, cond_embedded)
    cell = decoder.init_cell()
    token = torch.LongTensor([sos_token]).to(device)

    if tf_r is None:
        tf = False
    else:
        tf = True if random.random() < tf_r else False

    outputs = []
    for i in range(max_len):
        output, hidden, cell = decoder(token, hidden, cell)
        outputs.append(output)
        output_token = torch.max(torch.softmax(output, dim=1), 1)[1]
        if tf:
            token = teacher_token[i+1]
        else:
            if output_token.item() == eos_token:
                break
            token = output_token

    if len(outputs) != 0:
        outputs = torch.cat(outputs, dim=0).to(device)
    else:
        outputs = torch.FloatTensor([]).reshape(0, 28).to(device)
    return outputs


def to_long(array):
    return torch.LongTensor(array)


def train(
        encoder, decoder, tokenizer, train_loader, test_loader,
        latent_size, start_epoch, num_epoch, lr,
        task_name, log_dir, cpt_dir,
        tf_ratio, final_tf_r, kld_weight, final_kld_w, kld_anneal_cycle,
        eval_interval):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(os.path.join(log_dir, task_name))

    cpt_dir = os.path.join(cpt_dir, task_name)
    os.makedirs(cpt_dir, exist_ok=True)

    sos_token = tokenizer.char2token['SOS']
    eos_token = tokenizer.char2token['EOS']

    metrics = []
    num_iter = 0
    total_loss = 0
    total_kld_loss = 0
    total_ce_loss = 0

    epoch_pbar = tqdm(range(start_epoch, num_epoch))
    for epoch in epoch_pbar:
        tf_r = tf_ratio(epoch, num_epoch, final_tf_r) if callable(tf_ratio) else tf_ratio
        kld_w = kld_weight(epoch, num_epoch, final_kld_w, kld_anneal_cycle) if callable(kld_weight) else kld_weight

        iter_pbar = tqdm(train_loader)
        for idx, (word, cond) in enumerate(iter_pbar):
            encoder.train()
            decoder.train()

            word = word[0]
            cond = cond[0]
            token = to_long(tokenizer.tokenize(word)).to(device)
            cond = to_long([cond]).to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # encode + reparameterization
            hidden = encoder.init_hidden()
            cell = encoder.init_cell()
            m, lv, z = encoder(token[1:], hidden, cell, cond)

            # decode
            cond_embedded = encoder.embed_cond(cond).reshape(1, 1, -1).to(device)
            output = inference(decoder, z, cond_embedded, token, tf_r, sos_token, eos_token, token.shape[0]-1)

            output_length = output.shape[0]
            reconstruction = criterion(output, token[1:1+output_length].to(device))
            regularization = kld_loss(m, lv)

            loss = reconstruction + kld_w * regularization
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            total_ce_loss += reconstruction.item()
            total_kld_loss += regularization.item()

            if torch.isnan(loss) or torch.isnan(regularization):
                # raise AttributeError(f'Loss became nan. Total loss: {loss.item()}, KLD loss : {regularization.item()}')
                print(f'Loss became nan. Total loss: {loss.item()}, KLD loss : {regularization.item()}')

            iter_pbar.set_description(f'[train: {epoch+1}/{num_epoch}]')
        
            num_iter += 1

            if num_iter%eval_interval == 0:
                encoder.eval()
                decoder.eval()

                avg_loss = total_loss / eval_interval
                avg_ce_loss = total_ce_loss / eval_interval
                avg_kld_loss = total_kld_loss / eval_interval
                total_bleu = 0
                total_g_score = 0
                for i in range(5):
                    bleu, g_score = evaluation(encoder, decoder, tokenizer, test_loader, latent_size)
                    total_bleu += bleu
                    total_g_score += g_score
                avg_bleu = total_bleu / 5
                avg_g_score = total_g_score / 5
                metrics.append([avg_loss, avg_ce_loss, avg_kld_loss, tf_r, kld_w, avg_bleu, avg_g_score])
                epoch_pbar.set_description(f'[train: {epoch+1}/{num_epoch}] ce: {avg_ce_loss:.3f}, kld: {avg_kld_loss:.3f}, bleu-4: {avg_bleu:.3f}, g-score: {avg_g_score:.3f}')

                scalars = {
                    'Loss': avg_loss,
                    'CrossEntropy': avg_ce_loss,
                    'KLD': avg_kld_loss,
                    'Teacher ratio': tf_r,
                    'KLD weight': kld_w,
                    'BLEU4 score': avg_bleu,
                    'Gaussian score': avg_g_score
                }
                logger.add_scalars('train', scalars, num_iter//eval_interval)

                torch.save(encoder.state_dict(), os.path.join(cpt_dir, f'iter{num_iter}_encoder.cpt'))
                torch.save(decoder.state_dict(), os.path.join(cpt_dir, f'iter{num_iter}_decoder.cpt'))

                total_loss = 0
                total_kld_loss = 0
                total_ce_loss = 0

    return metrics


def evaluation(encoder, decoder, tokenizer, loader, latent_size):
    sos_token = tokenizer.char2token['SOS']
    eos_token = tokenizer.char2token['EOS']

    bleu_score = []
    with torch.no_grad():
        for idx, data in enumerate(loader):
            if loader.dataset.mode == 'train':
                src_word, src_cond = data
                dest_word, dest_cond = data
            else:
                src_word, src_cond, dest_word, dest_cond = data

            src_word = src_word[0]
            src_cond = src_cond[0]
            dest_word = dest_word[0]
            dest_cond = dest_cond[0]

            src_token = to_long(tokenizer.tokenize(src_word)).to(device)
            src_cond = to_long([src_cond]).to(device)
            dest_token = to_long(tokenizer.tokenize(dest_word)).to(device)
            dest_cond = to_long([dest_cond]).to(device)

            hidden = encoder.init_hidden()
            cell = encoder.init_cell()
            m, lv, z = encoder(src_token[1:], hidden, cell, src_cond)

            dest_cond_embedded = encoder.embed_cond(dest_cond).reshape(1, 1, -1).to(device)
            output = inference(decoder, z, dest_cond_embedded, None, None, sos_token, eos_token, dest_token.shape[0]-1)

            output_token = torch.max(torch.softmax(output, dim=1), 1)[1]
            output_word = tokenizer.inv_tokenize(output_token)
            src_word = tokenizer.inv_tokenize(src_token)
            dest_word = tokenizer.inv_tokenize(dest_token)
            bleu_score.append(compute_bleu(output_word, dest_word))

    words = generate_words(encoder, decoder, latent_size, tokenizer)
    g_score = gaussian_score(words)

    return sum(bleu_score)/len(bleu_score), g_score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_size', type=int, default=28)
    parser.add_argument('--hidden_input_size', type=int, default=256)
    parser.add_argument('--cond_size', type=int, default=4)
    parser.add_argument('--hidden_cond_size', type=int, default=8)
    parser.add_argument('--latent_size', type=int, default=32)

    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.05)

    parser.add_argument('--tf_ratio', type=float)
    parser.add_argument('--final_tf_r', type=float, default=0.5)

    parser.add_argument('--kld_weight', type=float)
    parser.add_argument('--final_kld_w', type=float, default=0.4)
    parser.add_argument('--kld_anneal_cycle', type=int, default=2)

    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--cpt_dir', type=str, default='cpts')

    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))

    input_size = args.input_size
    hidden_input_size = args.hidden_input_size
    cond_size = args.cond_size
    hidden_cond_size = args.hidden_cond_size
    latent_size = args.latent_size

    num_epoch = args.num_epoch
    start_epoch = args.start_epoch
    eval_interval = args.eval_interval
    lr = args.lr

    tf_ratio = teacher_forcing_ratio if args.tf_ratio is None else args.tf_ratio
    final_tf_r = args.final_tf_r

    kld_weight = kld_weight_annealing if args.kld_weight is None else args.kld_weight
    final_kld_w = args.final_kld_w
    kld_anneal_cycle = args.kld_anneal_cycle

    log_dir = args.log_dir
    cpt_dir = args.cpt_dir

    task_name = f'in{input_size}_h-in{hidden_input_size}_cond{cond_size}_h-cond{hidden_cond_size}_l{latent_size}'
    task_name += f'_start{start_epoch}-{num_epoch}epoch_eval{eval_interval}_lr{lr}'
    if callable(tf_ratio):
        task_name += f'_f-tfr{final_tf_r}'
    else:
        task_name += f'_tfr{tf_ratio}'

    if callable(kld_weight):
        task_name += f'_final-kldw{final_kld_w}_{kld_anneal_cycle}cycle'
    else:
        task_name += f'_kldw{kld_weight}'

    train_dataset = WordDataset('.', 'train')
    test_dataset = WordDataset('.', 'test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    encoder = EncoderRNN(input_size, hidden_input_size, cond_size, hidden_cond_size, latent_size).to(device)
    decoder = DecoderRNN(input_size, hidden_input_size, hidden_cond_size, latent_size).to(device)
    tokenizer = OneHotEncoder()
    train(
        encoder, decoder, tokenizer, train_loader, test_loader,
        latent_size, start_epoch, num_epoch, lr,
        task_name, log_dir, cpt_dir,
        tf_ratio, final_tf_r, kld_weight, final_kld_w, kld_anneal_cycle,
        eval_interval)

    if callable(tf_ratio):
        tf_r = []
        num_iter = 0
        for i in range(num_epoch):
            for j in range(len(train_loader)):
                num_iter += 1
                if num_iter%eval_interval == 0:
                    tf_r.append(tf_ratio(i, num_epoch, final_tf_r))
        plt.plot(np.arange(len(tf_r)), tf_r, label='tfr')

    if callable(kld_weight):
        kld_w = []
        num_iter = 0
        for i in range(num_epoch):
            for j in range(len(train_loader)):
                num_iter += 1
                if num_iter%eval_interval == 0:
                    kld_w.append(kld_weight(i, num_epoch, final_kld_w, kld_anneal_cycle))
        plt.plot(np.arange(len(kld_w)), kld_w, label='kld anneal')

    if callable(tf_ratio) or callable(kld_weight):
        plt.legend()
        plt.grid()
        plt.savefig(f'{task_name}.png', dpi=300)
