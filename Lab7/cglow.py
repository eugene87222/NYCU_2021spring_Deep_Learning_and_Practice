# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn

import cglow_block

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CondGlowStep(nn.Module):
    def __init__(self, input_sz, cond_sz, cond_conv_chs, cond_fc_chs, affine_conv_chs):
        super(CondGlowStep, self).__init__()
        self.cond_actnorm = cglow_block.CondActnorm(input_sz=input_sz, cond_sz=cond_sz, cond_conv_chs=cond_conv_chs, cond_fc_chs=cond_fc_chs)
        self.cond_conv1x1 = cglow_block.CondConv1x1(input_sz=input_sz, cond_sz=cond_sz, cond_conv_chs=cond_conv_chs, cond_fc_chs=cond_fc_chs)
        self.cond_affine_coupling = cglow_block.CondAffineCoupling(input_sz=[input_sz[0]//2, *input_sz[1:]], cond_sz=cond_sz, affine_conv_chs=affine_conv_chs)

    def forward(self, x, cond, log_det=None, reverse=False):
        if reverse is False:
            x, log_det = self.cond_actnorm(x, cond, log_det, reverse=False)
            x, log_det = self.cond_conv1x1(x, cond, log_det, reverse=False)
            x, log_det = self.cond_affine_coupling(x, cond, log_det, reverse=False)
        else:
            x, log_det = self.cond_affine_coupling(x, cond, log_det, reverse=True)
            x, log_det = self.cond_conv1x1(x, cond, log_det, reverse=True)
            x, log_det = self.cond_actnorm(x, cond, log_det, reverse=True)
        return x, log_det


class CondGlow(nn.Module):
    def __init__(
            self, input_sz, cond_sz, cond_conv_chs, cond_fc_chs,
            affine_conv_chs, flow_depth, num_levels):
        super(CondGlow, self).__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.flow_depth = flow_depth
        self.num_levels = num_levels

        c, h, w = input_sz
        for l in range(num_levels):
            # Squeeze Layer
            c, h, w = c*4, h//2, w//2
            input_sz = (c, h, w)
            self.layers.append(cglow_block.SqueezeLayer(factor=2))
            self.output_shapes.append((-1, c, h, w))

            # K Conditional Glow Step
            for k in range(flow_depth):
                self.layers.append(
                    CondGlowStep(
                        input_sz=input_sz,
                        cond_sz=cond_sz,
                        cond_conv_chs=cond_conv_chs,
                        cond_fc_chs=cond_fc_chs,
                        affine_conv_chs=affine_conv_chs))
                self.output_shapes.append((-1, c, h, w))

            # Split
            if l < num_levels - 1:
                self.layers.append(cglow_block.Split2d(num_chs=c))
                self.output_shapes.append((-1, c//2, h, w))
                c = c // 2

    def forward(self, x, cond, log_det=0, reverse=False):
        if not reverse:
            return self.encode(x, cond, log_det)
        else:
            return self.decode(x, cond, log_det)

    def encode(self, x, cond, log_det=0):
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, cglow_block.Split2d) or isinstance(layer, cglow_block.SqueezeLayer):
                x, log_det = layer(x, log_det, reverse=False)
            else:
                x, log_det = layer(x, cond, log_det, reverse=False)
        return x, log_det

    def decode(self, x, cond, log_det=0):
        for layer in reversed(self.layers):
            if isinstance(layer, cglow_block.Split2d):
                x, log_det = layer(x, log_det=log_det, reverse=True)
            elif isinstance(layer, cglow_block.SqueezeLayer):
                x, log_det = layer(x, log_det=log_det, reverse=True)
            else:
                x, log_det = layer(x, cond, log_det=log_det, reverse=True)
        return x, log_det


class CondGlowModel(nn.Module):
    def __init__(self, args):
        super(CondGlowModel, self).__init__()
        self.cond_sz = args.cond_sz
        self.embed_c= nn.Sequential(
            nn.Linear(args.num_conditions, args.cond_sz[0]*args.cond_sz[1]*args.cond_sz[2]),
            nn.ReLU(inplace=True))

        self.flow = CondGlow(
                        input_sz=args.input_sz,
                        cond_sz=args.cond_sz,
                        cond_conv_chs=args.cond_conv_chs,
                        cond_fc_chs=args.cond_fc_chs,
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

    def forward(self, x, cond, reverse=False):
        cond = self.embed_c(cond)
        cond = cond.reshape(-1, self.cond_sz[0], self.cond_sz[1], self.cond_sz[2])
        if not reverse:
            dims = x.shape[1] * x.shape[2] * x.shape[3]
            log_det = torch.zeros(x.shape[0], device=device)
            z, objective = self.flow(x, cond, log_det=log_det, reverse=False)
            mean, log_std = self.prior()
            objective += cglow_block.gaussian_likelihood(z, mean, log_std)
            nll = -objective / (np.log(2)*dims)
            return z, nll
        else:
            with torch.no_grad():
                mean, log_std = self.prior()
                if x is None:
                    x = cglow_block.batch_gaussian_sample(cond.shape[0], mean, log_std)
                x, log_det = self.flow(x, cond, reverse=True)
            return x, log_det
