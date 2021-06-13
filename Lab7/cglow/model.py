# -*- coding: utf-8 -*-
import numpy as np
from math import log, pi, exp
from scipy import linalg as la

import torch
from torch import nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gaussian_log_p(x, mean, log_std):
    return -0.5*log(2*pi) - log_std - 0.5*(x-mean)**2/torch.exp(2*log_std)


def gaussian_sample(eps, mean, log_std):
    return mean + torch.exp(log_std)*eps


class CondActnorm(nn.Module):
    def __init__(self, in_chs, cond_sz, cond_fc_fts):
        super(CondActnorm, self).__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(cond_sz, cond_fc_fts),
            nn.ReLU(inplace=True),
            nn.Linear(cond_fc_fts, cond_fc_fts),
            nn.ReLU(inplace=True),
            nn.Linear(cond_fc_fts, 2*in_chs),
            nn.Tanh())
        self.cond_net[0].weight.data.zero_()
        self.cond_net[0].bias.data.zero_()
        self.cond_net[2].weight.data.zero_()
        self.cond_net[2].bias.data.zero_()
        self.cond_net[4].weight.data.zero_()
        self.cond_net[4].bias.data.zero_()

    def forward(self, x, cond):
        cond_b, _, = cond.shape
        cond = self.cond_net(cond)
        cond = cond.reshape(cond_b, -1, 1, 1)
        log_scale, bias = cond.chunk(2, dim=1)
        dims = x.shape[2] * x.shape[3]
        x = x + bias
        x = x * torch.exp(log_scale)
        dlog_det = torch.sum(log_scale, dim=(1, 2, 3)) * dims
        return x, dlog_det

    def reverse(self, x, cond):
        cond_b, _, = cond.shape
        cond = self.cond_net(cond)
        cond = cond.reshape(cond_b, -1, 1, 1)
        log_scale, bias = cond.chunk(2, dim=1)
        dims = x.shape[2] * x.shape[3]
        x = x * torch.exp(-log_scale)
        x = x - bias
        dlog_det = -torch.sum(log_scale, dim=(1, 2, 3)) * dims
        return x, dlog_det


class CondInvertible1x1Conv(nn.Module):
    def __init__(self, in_chs, cond_sz, cond_fc_fts):
        super(CondInvertible1x1Conv, self).__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(cond_sz, cond_fc_fts),
            nn.ReLU(inplace=True),
            nn.Linear(cond_fc_fts, cond_fc_fts),
            nn.ReLU(inplace=True),
            nn.Linear(cond_fc_fts, in_chs*in_chs))
        self.cond_net[0].weight.data.zero_()
        self.cond_net[0].bias.data.zero_()
        self.cond_net[2].weight.data.zero_()
        self.cond_net[2].bias.data.zero_()
        self.cond_net[4].weight.data.normal_(0, 0.05)
        self.cond_net[4].bias.data.zero_()

    def get_weight(self, x, cond, inverse=False):
        x_c = x.shape[1]
        cond_b, _ = cond.shape
        cond = self.cond_net(cond)
        cond = torch.tanh(cond)
        weight = cond.reshape(cond_b, x_c, x_c)
        dims = x.shape[2] * x.shape[3]
        dlog_det = torch.slogdet(weight)[1] * dims
        if inverse:
            weight = torch.inverse(weight)
        weight = weight.reshape(cond_b, x_c, x_c, 1, 1)
        return weight, dlog_det

    def forward(self, x, cond):
        weight, dlog_det = self.get_weight(x, cond)
        x_b, x_c, x_h, x_w = x.shape
        x = x.reshape(1, x_b*x_c, x_h, x_w)
        w_b, w_c, _, w_h, w_w = weight.shape
        assert x_b==w_b and x_c==w_c, 'The input and kernel dimensions are different'
        weight = weight.reshape(w_b*w_c, w_c, w_h, w_w)
        z = F.conv2d(x, weight, groups=x_b)
        z = z.reshape(x_b, x_c, x_h, x_w)
        return z, dlog_det

    def reverse(self, x, cond):
        weight, dlog_det = self.get_weight(x, cond, inverse=True)
        x_b, x_c, x_h, x_w = x.shape
        x = x.reshape(1, x_b*x_c, x_h, x_w)
        w_b, w_c, _, w_h, w_w = weight.shape
        assert x_b==w_b and x_c==w_c, 'The input and kernel dimensions are different'
        weight = weight.reshape(w_b*w_c, w_c, w_h, w_w)
        z = F.conv2d(x, weight, groups=x_b)
        z = z.reshape(x_b, x_c, x_h, x_w)
        return z


class ZeroConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, padding=1):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.log_scale = nn.Parameter(torch.zeros(1, out_chs, 1, 1))

    def forward(self, x):
        out = F.pad(x, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.log_scale*3)
        return out


class CondAffineCoupling(nn.Module):
    def __init__(self, in_sz, cond_sz, cond_fc_fts, affine_conv_chs):
        super(CondAffineCoupling, self).__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(cond_sz, cond_fc_fts),
            nn.ReLU(inplace=True),
            nn.Linear(cond_fc_fts, cond_fc_fts),
            nn.ReLU(inplace=True),
            nn.Linear(cond_fc_fts, in_sz[0]*in_sz[1]*in_sz[2]),
            nn.ReLU(inplace=True))
        self.cond_net[0].weight.data.zero_()
        self.cond_net[0].bias.data.zero_()
        self.cond_net[2].weight.data.zero_()
        self.cond_net[2].bias.data.zero_()
        self.cond_net[4].weight.data.zero_()
        self.cond_net[4].bias.data.zero_()
        self.net = nn.Sequential(
            nn.Conv2d(in_sz[0]*2, affine_conv_chs, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(affine_conv_chs, affine_conv_chs, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            ZeroConv2d(affine_conv_chs, 2*in_sz[0]))
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x, cond):
        z1, z2 = x.chunk(2, dim=1)
        cond = self.cond_net(cond)
        cond = cond.reshape(cond.shape[0], x.shape[1]//2, *x.shape[2:])
        tmp = torch.cat((z1, cond), dim=1)
        tmp = self.net(tmp)
        log_scale, shift = tmp.chunk(2, dim=1)
        scale = torch.sigmoid(log_scale+2)
        z2 = (z2+shift) * scale
        dlog_det = torch.sum(torch.log(scale), dim=(1, 2, 3))
        z = torch.cat((z1, z2), dim=1)
        return z, dlog_det

    def reverse(self, x, cond):
        z1, z2 = x.chunk(2, dim=1)
        cond = self.cond_net(cond)
        cond = cond.reshape(cond.shape[0], x.shape[1]//2, *x.shape[2:])
        tmp = torch.cat((z1, cond), dim=1)
        tmp = self.net(tmp)
        log_scale, shift = tmp.chunk(2, dim=1)
        scale = torch.sigmoid(log_scale+2)
        z2 = z2/scale - shift
        z = torch.cat((z1, z2), dim=1)
        return z


class SqueezeLayer(nn.Module):
    def __init__(self):
        super(SqueezeLayer, self).__init__()

    def squeeze(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h//2, 2, w//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(b, c*4, h//2, w//2)
        return x

    def unsqueeze(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c//4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(b, c//4, h*2, w*2)
        return x


class CondGlowStep(nn.Module):
    def __init__(self, in_sz, cond_sz, cond_fc_fts, affine_conv_chs):
        super(CondGlowStep, self).__init__()
        self.actnorm = CondActnorm(in_sz[0], cond_sz, cond_fc_fts)
        self.conv1x1 = CondInvertible1x1Conv(in_sz[0], cond_sz, cond_fc_fts)
        self.affine_coupling = CondAffineCoupling(in_sz[0], cond_sz, cond_fc_fts, affine_conv_chs)

    def forward(self, x, cond):
        out, dlog_det1 = self.actnorm.forward(x, cond)
        out, dlog_det2 = self.conv1x1.forward(out, cond)
        out, dlog_det3 = self.affine_coupling.forward(out, cond)
        log_det = dlog_det1 + dlog_det2 + dlog_det3
        return out, log_det

    def reverse(self, x, cond):
        out = self.affine_coupling.reverse(x, cond)
        out = self.conv1x1.reverse(out, cond)
        out = self.actnorm.reverse(out, cond)
        return out


class CondGlowBlock(nn.Module):
    def __init__(self, in_sz, flow_depth, cond_sz, cond_fc_fts, affine_conv_chs, split=True):
        super(CondGlowBlock, self).__init__()
        squeeze_dim = in_sz[0] * 4
        self.squeeze = SqueezeLayer()
        self.cglows = nn.ModuleList()
        for i in range(flow_depth):
            self.cglows.append(CondGlowStep(
                in_sz=(squeeze_dim, in_sz[1]//2, in_sz[2]//2),
                cond_sz=cond_sz,
                cond_fc_fts=cond_fc_fts,
                affine_conv_chs=affine_conv_chs))
        self.split = split
        if split:
            self.prior = ZeroConv2d(in_sz[0]*2, in_sz[0]*4)
        else:
            self.prior = ZeroConv2d(in_sz[0]*4, in_sz[0]*8)

    def forward(self, x, cond):
        b, c, h, w = x.shape
        out = self.squeeze.squeeze(x)
        log_det = 0
        for cglow in self.cglows:
            out, dlog_det = cglow.forward(out, cond)
            log_det = log_det + dlog_det

        if self.split:
            z1, z2 = out.chunk(2, dim=1)
            mean, log_std = self.prior(z1).chunk(2, dim=1)
            log_p = gaussian_log_p(z2, mean, log_std).sum(dim=(1, 2, 3))
            out = z1
        else:
            mean, log_std = self.prior(torch.zeros_like(out)).chunk(2, dim=1)
            log_p = gaussian_log_p(out, mean, log_std).sum(dim=(1, 2, 3))

        return out, log_det, log_p

    def reverse(self, x, cond, eps=None, reconstruct=False):
        z1 = x
        if reconstruct:
            if self.split:
                z1 = torch.cat([x, eps], dim=1)
            else:
                z1 = eps
        else:
            if self.split:
                mean, log_std = self.prior(z1).chunk(2, dim=1)
                z2 = gaussian_sample(eps, mean, log_std)
                z = torch.cat([z1, z2], dim=1)
            else:
                mean, log_std = self.prior(torch.zeros_like(z1)).chunk(2, dim=1)
                z = gaussian_sample(eps, mean, log_std)

        for glow in self.glows[::-1]:
            z = glow.reverse(z, cond)
        unsqueezed = self.squeeze.unsqueeze(z)
        return unsqueezed


class CondGlow(nn.Module):
    def __init__(self, in_sz, flow_depth, num_levels, cond_sz, cond_fc_fts, affine_conv_chs):
        super(CondGlow, self).__init__()
        self.cglow_blocks = nn.ModuleList()
        for i in range(num_levels-1):
            self.cglow_blocks.append(CondGlowBlock(
                                        in_sz=in_sz,
                                        flow_depth=flow_depth,
                                        cond_sz=cond_sz,
                                        cond_fc_fts=cond_fc_fts,
                                        affine_conv_chs=affine_conv_chs,
                                        split=True))
            in_sz = [in_sz[0]*2, in_sz[1]//2, in_sz[2]//2]
        self.cglow_blocks.append(CondGlowBlock(
                                    in_sz=in_sz,
                                    flow_depth=flow_depth,
                                    cond_sz=cond_sz,
                                    cond_fc_fts=cond_fc_fts,
                                    affine_conv_chs=affine_conv_chs,
                                    split=False))

    def forward(self, x, cond):
        log_p_sum = 0
        log_det = 0
        out = x
        for block in self.cglow_blocks:
            out, dlog_det, log_p = block.forward(out, cond)
            log_det = log_det + dlog_det
            log_p_sum = log_p_sum + log_p
        return log_p_sum, log_det

    def reverse(self, zs, cond, reconstruct=False):
        for idx, block in enumerate(self.glow_blocks[::-1]):
            if idx == 0:
                out = block.reverse(zs[-1], cond, zs[-1], reconstruct=reconstruct)
            else:
                out = block.reverse(out, cond, zs[-1-idx], reconstruct=reconstruct)
        return out
