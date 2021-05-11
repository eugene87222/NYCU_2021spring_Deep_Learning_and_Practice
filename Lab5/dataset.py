# -*- coding: utf-8 -*-
import os
import numpy as np

from torch.utils.data import Dataset


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