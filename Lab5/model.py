# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        self.mean_hidden= nn.Linear(hidden_input_size, latent_size)
        self.logvar_hidden = nn.Linear(hidden_input_size, latent_size)
        self.mean_cell = nn.Linear(hidden_input_size, latent_size)
        self.logvar_cell = nn.Linear(hidden_input_size, latent_size)

    def forward(self, inputs, hidden, cell, cond):
        cond_embedded = self.embed_cond(cond).reshape(1, 1, -1)
        hidden = torch.cat((hidden, cond_embedded), dim=2)
        cell = torch.cat((cell, cond_embedded), dim=2)

        inputs_embedded = self.embed_word(inputs).reshape(-1, 1, self.hidden_input_size)
        outputs, (hidden, cell) = self.lstm(inputs_embedded, (hidden, cell))

        m_hidden = self.mean_hidden(hidden)
        lv_hidden = self.logvar_hidden(hidden)
        epsilon_hidden = torch.normal(torch.zeros(self.latent_size), torch.ones(self.latent_size)).to(device)
        z_hidden = torch.exp(lv_hidden/2)*epsilon_hidden + m_hidden

        m_cell = self.mean_cell(cell)
        lv_cell = self.logvar_cell(cell)
        epsilon_cell = torch.normal(torch.zeros(self.latent_size), torch.ones(self.latent_size)).to(device)
        z_cell = torch.exp(lv_cell/2)*epsilon_cell + m_cell

        return m_hidden, lv_hidden, z_hidden, m_cell, lv_cell, z_cell

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_input_size-self.hidden_cond_size, device=device)

    def init_cell(self):
        return torch.zeros(1, 1, self.hidden_input_size-self.hidden_cond_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_input_size, hidden_cond_size, latent_size):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_input_size = hidden_input_size
        self.hidden_cond_size = hidden_cond_size
        self.latent_size = latent_size

        self.latent_hidden_to_hidden = nn.Linear(latent_size+hidden_cond_size, hidden_input_size)
        self.latent_cell_to_hidden = nn.Linear(latent_size+hidden_cond_size, hidden_input_size)
        self.embed_word = nn.Embedding(input_size, hidden_input_size)
        self.lstm = nn.LSTM(hidden_input_size, hidden_input_size)
        self.out = nn.Linear(hidden_input_size, input_size)

    def forward(self, inputs, hidden, cell):
        inputs_embedded = self.embed_word(inputs).reshape(-1, 1, self.hidden_input_size)
        inputs_embedded = F.relu(inputs_embedded)
        output, (hidden, cell) = self.lstm(inputs_embedded, (hidden, cell))
        output = self.out(output).reshape(-1, self.input_size)
        return output, hidden, cell

    def init_hidden(self, z_hidden, cond_embedded):
        latent_hidden = torch.cat((z_hidden, cond_embedded), dim=2)
        return self.latent_hidden_to_hidden(latent_hidden)

    def init_cell(self, z_cell, cond_embedded):
        latent_cell = torch.cat((z_cell, cond_embedded), dim=2)
        return self.latent_cell_to_hidden(latent_cell)
