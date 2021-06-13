# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn

import glow.glow_block as glow_block

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GlowStep(nn.Module):
    def __init__(self, input_sz, affine_conv_chs):
        super(GlowStep, self).__init__()
        self.actnorm = glow_block.Actnorm(num_chs=input_sz[0])
        self.conv1x1 = glow_block.Conv1x1(num_chs=input_sz[0])
        self.affine_coupling = glow_block.AffineCoupling(input_sz=[input_sz[0]//2, *input_sz[1:]], affine_conv_chs=affine_conv_chs)

    def forward(self, x, log_det=None, reverse=False):
        if reverse is False:
            x, log_det = self.actnorm(x, log_det, reverse=False)
            x, log_det = self.conv1x1(x, log_det, reverse=False)
            x, log_det = self.affine_coupling(x, log_det, reverse=False)
        else:
            x, log_det = self.affine_coupling(x, log_det, reverse=True)
            x, log_det = self.conv1x1(x, log_det, reverse=True)
            x, log_det = self.actnorm(x, log_det, reverse=True)
        return x, log_det


class Glow(nn.Module):
    def __init__(self, input_sz, affine_conv_chs, flow_depth, num_levels):
        super(Glow, self).__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.flow_depth = flow_depth
        self.num_levels = num_levels

        c, h, w = input_sz
        for l in range(num_levels):
            # Squeeze Layer
            c, h, w = c*4, h//2, w//2
            input_sz = (c, h, w)
            self.layers.append(glow_block.SqueezeLayer(factor=2))
            self.output_shapes.append((-1, c, h, w))

            # K Glow Step
            for k in range(flow_depth):
                self.layers.append(
                    GlowStep(
                        input_sz=input_sz,
                        affine_conv_chs=affine_conv_chs))
                self.output_shapes.append((-1, c, h, w))

            # Split
            if l < num_levels - 1:
                self.layers.append(glow_block.Split2d(num_chs=c))
                self.output_shapes.append((-1, c//2, h, w))
                c = c // 2

    def forward(self, x, log_det=0, reverse=False):
        if not reverse:
            return self.encode(x, log_det)
        else:
            return self.decode(x, log_det)

    def encode(self, x, log_det=0):
        for layer, shape in zip(self.layers, self.output_shapes):
            x, log_det = layer(x, log_det, reverse=False)
        return x, log_det

    def decode(self, x, cond, log_det=0):
        for layer in reversed(self.layers):
                x, log_det = layer(x, log_det=log_det, reverse=True)
        return x, log_det


class GlowModel(nn.Module):
    def __init__(self, args):
        super(GlowModel, self).__init__()

        self.flow = Glow(
                        input_sz=args.input_sz,
                        affine_conv_chs=args.affine_conv_chs,
                        flow_depth=args.flow_depth,
                        num_levels=args.num_levels)
        self.learn_prior = args.learn_prior
        self.register_parameter(
            'new_mean',
            nn.Parameter(torch.zeros([
                1,
                self.flow.output_shapes[-1][1],
                self.flow.output_shapes[-1][2],
                self.flow.output_shapes[-1][3]
            ])))
        self.register_parameter(
            'new_log_std',
            nn.Parameter(torch.zeros([
                1,
                self.flow.output_shapes[-1][1],
                self.flow.output_shapes[-1][2],
                self.flow.output_shapes[-1][3]
            ])))

    def prior(self):
        if self.learn_prior:
            return self.new_mean, self.new_log_std
        else:
            return torch.zeros(self.new_mean.shape, device=device), torch.zeros(self.new_log_std.shape, device=device)

    def forward(self, x, bs=5, reverse=False):
        if not reverse:
            dims = x.shape[1] * x.shape[2] * x.shape[3]
            log_det = torch.zeros(x.shape[0], device=device)
            z, objective = self.flow(x, log_det=log_det, reverse=False)
            mean, log_std = self.prior()
            objective += glow_block.log_likelihood(z, mean, log_std)
            nll = -objective / (np.log(2)*dims)
            return z, nll
        else:
            with torch.no_grad():
                mean, log_std = self.prior()
                if x is None:
                    x = glow_block.batchsample(bs, mean, log_std)
                x, log_det = self.flow(x, reverse=True)
            return x, log_det
