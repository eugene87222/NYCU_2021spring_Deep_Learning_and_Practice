# -*- coding: utf-8 -*-
import numpy as np
from math import log, pi, exp
from scipy import linalg as la

import torch
from torch import nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_abs = lambda x: torch.log(torch.abs(x))


def gaussian_log_p(x, mean, log_std):
    return -0.5*log(2*pi) - log_std - 0.5*(x-mean)**2/torch.exp(2*log_std)


def gaussian_sample(eps, mean, log_std):
    return mean + torch.exp(log_std)*eps


class Actnorm(nn.Module):
    def __init__(self, in_chs, actnorm_inited):
        super(Actnorm, self).__init__()
        size = [1, in_chs, 1, 1]
        self.bias = nn.Parameter(torch.zeros(size))
        self.log_scale = nn.Parameter(torch.zeros(size))
        self.inited = actnorm_inited

    def init(self, x):
        if not self.training:
            raise ValueError('In eval() mode, but Actnorm not initialized')
        with torch.no_grad():
            flatten = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            log_scale = torch.log(1/(std+1e-6))
            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(log_scale)
            self.inited = True

    def forward(self, x):
        if not self.inited:
            self.init(x)
        dims = x.shape[2] * x.shape[3]
        x = x + self.bias
        x = x * torch.exp(self.log_scale)
        dlog_det = torch.sum(self.log_scale) * dims
        return x, dlog_det

    def reverse(self, x):
        x = x * torch.exp(-self.log_scale)
        x = x - self.bias
        return x


class Invertible1x1Conv(nn.Module):
    def __init__(self, in_chs):
        super(Invertible1x1Conv, self).__init__()
        weight = np.random.randn(in_chs, in_chs)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask.copy()))
        self.register_buffer('l_mask', torch.from_numpy(l_mask.copy()))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(log_abs(w_s))
        self.w_u = nn.Parameter(w_u)

    def get_weight(self):
        weight = (
            self.w_p
            @ (self.w_l*self.l_mask+self.l_eye)
            @ ((self.w_u*self.u_mask)+torch.diag(self.s_sign*torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        weight = self.get_weight()
        z = F.conv2d(x, weight)
        dlog_det = torch.sum(self.w_s) * x.shape[2] * x.shape[3]
        return z, dlog_det

    def reverse(self, x):
        weight = self.get_weight()
        weight = weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        z = F.conv2d(x, weight)
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


class AffineCoupling(nn.Module):
    def __init__(self, in_chs, affine_conv_chs=512):
        super(AffineCoupling, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chs//2, affine_conv_chs, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(affine_conv_chs, affine_conv_chs, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            ZeroConv2d(affine_conv_chs, in_chs))
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x):
        z1, z2 = x.chunk(2, dim=1)
        log_scale, shift = self.net(z1).chunk(2, dim=1)
        scale = torch.sigmoid(log_scale+2)
        z2 = (z2+shift) * scale
        dlog_det = torch.sum(torch.log(scale), dim=(1, 2, 3))
        z = torch.cat((z1, z2), dim=1)
        return z, dlog_det

    def reverse(self, x):
        z1, z2 = x.chunk(2, dim=1)
        log_scale, shift = self.net(z1).chunk(2, dim=1)
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


class GlowStep(nn.Module):
    def __init__(self, in_chs, affine_conv_chs, actnorm_inited=False):
        super(GlowStep, self).__init__()
        self.actnorm = Actnorm(in_chs, actnorm_inited)
        self.conv1x1 = Invertible1x1Conv(in_chs)
        self.affine_coupling = AffineCoupling(in_chs, affine_conv_chs)

    def forward(self, x):
        out, dlog_det1 = self.actnorm.forward(x)
        out, dlog_det2 = self.conv1x1.forward(out)
        out, dlog_det3 = self.affine_coupling.forward(out)
        log_det = dlog_det1 + dlog_det2 + dlog_det3
        return out, log_det

    def reverse(self, x):
        out = self.affine_coupling.reverse(x)
        out = self.conv1x1.reverse(out)
        out = self.actnorm.reverse(out)
        return out


class GlowBlock(nn.Module):
    def __init__(self, in_chs, flow_depth, affine_conv_chs, split=True, actnorm_inited=False):
        super(GlowBlock, self).__init__()
        squeeze_dim = in_chs * 4
        self.squeeze = SqueezeLayer()
        self.glows = nn.ModuleList()
        for i in range(flow_depth):
            self.glows.append(GlowStep(squeeze_dim, affine_conv_chs, actnorm_inited=actnorm_inited))
        self.split = split
        if split:
            self.prior = ZeroConv2d(in_chs*2, in_chs*4)
        else:
            self.prior = ZeroConv2d(in_chs*4, in_chs*8)

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.squeeze.squeeze(x)
        log_det = 0
        for glow in self.glows:
            out, dlog_det = glow.forward(out)
            log_det = log_det + dlog_det
        if self.split:
            z1, z2 = out.chunk(2, dim=1)
            mean, log_std = self.prior(z1).chunk(2, dim=1)
            log_p = gaussian_log_p(z2, mean, log_std).sum(dim=(1, 2, 3))
            out = z1
            z_new = z2
        else:
            mean, log_std = self.prior(torch.zeros_like(out)).chunk(2, dim=1)
            log_p = gaussian_log_p(out, mean, log_std).sum(dim=(1, 2, 3))
            z_new = out
        return out, log_det, log_p, z_new

    def reverse(self, x, eps=None, reconstruct=False):
        z1 = x
        if reconstruct:
            if self.split:
                z = torch.cat([x, eps], dim=1)
            else:
                z = eps
        else:
            if self.split:
                mean, log_std = self.prior(z1).chunk(2, dim=1)
                z2 = gaussian_sample(eps, mean, log_std)
                z = torch.cat([z1, z2], dim=1)
            else:
                mean, log_std = self.prior(torch.zeros_like(z1)).chunk(2, dim=1)
                z = gaussian_sample(eps, mean, log_std)

        for glow in self.glows[::-1]:
            z = glow.reverse(z)
        unsqueezed = self.squeeze.unsqueeze(z)
        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_chs, flow_depth, num_levels, affine_conv_chs, actnorm_inited=False):
        super(Glow, self).__init__()
        self.glow_blocks = nn.ModuleList()
        n_chs = in_chs
        for i in range(num_levels-1):
            self.glow_blocks.append(GlowBlock(n_chs, flow_depth, affine_conv_chs, split=True, actnorm_inited=actnorm_inited))
            n_chs *= 2
        self.glow_blocks.append(GlowBlock(n_chs, flow_depth, affine_conv_chs, split=False, actnorm_inited=actnorm_inited))

    def forward(self, x):
        log_p_sum = 0
        log_det = 0
        out = x
        z_outs = []
        for block in self.glow_blocks:
            out, dlog_det, log_p, z_new = block.forward(out)
            log_det = log_det + dlog_det
            log_p_sum = log_p_sum + log_p
            z_outs.append(z_new)
        return log_p_sum, log_det, z_outs

    def reverse(self, zs, reconstruct=False):
        for idx, block in enumerate(self.glow_blocks[::-1]):
            if idx == 0:
                out = block.reverse(zs[-1], zs[-1], reconstruct=reconstruct)
            else:
                out = block.reverse(out, zs[-1-idx], reconstruct=reconstruct)
        return out
