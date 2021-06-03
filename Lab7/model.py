# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, n_z, n_c, n_ch, add_bias=True):
        super(Generator, self).__init__()
        self.n_z = n_z
        self.n_c = n_c
        self.n_ch = n_ch
        self.add_bias = add_bias

        self.embed_c= nn.Sequential(
            nn.Linear(24, n_c),
            nn.ReLU(inplace=True))

        model = [
            nn.ConvTranspose2d(n_z+n_c, n_ch[0], kernel_size=4, stride=2, padding=0, bias=add_bias),
            nn.BatchNorm2d(n_ch[0]),
            nn.ReLU(inplace=True)
        ]
        for i in range(1, len(n_ch)):
            model += [
                nn.ConvTranspose2d(n_ch[i-1], n_ch[i], kernel_size=4, stride=2, padding=1, bias=add_bias),
                nn.BatchNorm2d(n_ch[i]),
                nn.ReLU(inplace=True)
            ]
        model += [
            nn.ConvTranspose2d(n_ch[-1], 3, kernel_size=4, stride=2, padding=1, bias=add_bias),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, z, c):
        z = z.view(-1, self.n_z, 1, 1)
        c_embd = self.embed_c(c).view(-1, self.n_c, 1, 1)
        x = torch.cat((z, c_embd), dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, img_h, img_w, n_ch, add_bias=True):
        super(Discriminator, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.n_ch = n_ch

        self.embed_c= nn.Sequential(
            nn.Linear(24, img_h*img_w),
            nn.LeakyReLU(0.2, inplace=True))

        model = [
            nn.Conv2d(4, n_ch[0], kernel_size=4, stride=2, padding=1, bias=add_bias),
            nn.BatchNorm2d(n_ch[0]),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for i in range(1, len(n_ch)):
            model += [
                nn.Conv2d(n_ch[i-1], n_ch[i], kernel_size=4, stride=2, padding=1, bias=add_bias),
                nn.BatchNorm2d(n_ch[i]),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        model += [
            nn.Conv2d(n_ch[-1], 1, kernel_size=4, stride=1, padding=0, bias=add_bias),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, image, c):
        c_embd = self.embed_c(c).view(-1, 1, self.img_h, self.img_w)
        x = torch.cat((image, c_embd), dim=1)
        return self.model(x).reshape(-1)
