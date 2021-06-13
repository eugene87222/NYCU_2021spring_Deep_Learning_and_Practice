# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_kernel_size(in_size, out_size, stride, padding):
    in_h, in_w = in_size[1:]
    out_h, out_w = out_size[1:]
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    k_h = in_h + 2*pad_h - (out_h-1)*stride_h
    k_w = in_w + 2*pad_w - (out_w-1)*stride_w
    return [k_h, k_w]


def log_gaussian(x, mean, log_std):
    tmp = 2*log_std + ((x-mean)**2)/torch.exp(2*log_std)
    return -0.5 * (tmp+np.log(2*np.pi))


def log_likelihood(x, mean, log_std):
    ll = log_gaussian(x, mean, log_std)
    return torch.sum(ll, dim=(1, 2, 3))


def sample(mean, log_std):
    z = torch.normal(mean, torch.exp(log_std))
    return z


def batchsample(bs, mean, log_std):
    z = sample(mean, log_std)
    for i in range(1, bs):
        z = torch.cat((z, sample(mean, log_std)), dim=0)
    return z


def split_feature(feature, mode='split'):
    # mode = ['split', 'cross']
    c = feature.shape[1]
    if mode == 'split':
        return feature[:,:c//2,...], feature[:,c//2:,...]
    elif mode == 'cross':
        return feature[:,0::2,...], feature[:,1::2,...]
    else:
        raise NotImplementedError()


class Linear(nn.Module):
    def __init__(self, in_features, out_features, init_mode='zero'):
        # init_mode = ['zero', 'normal']
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        if init_mode == 'zero':
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
        elif init_mode == 'normal':
            self.linear.weight.data.normal_(mean=0, std=0.1)
            self.linear.bias.data.normal_(mean=0, std=0.1)
        else:
            raise NotImplementedError()

    def forward(self, x):
        x = self.linear(x)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], init_mode='zero', add_actnorm=False):
        # init_mode = ['zero', 'normal']
        super(Conv2d, self).__init__()

        padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not add_actnorm)
        if init_mode == 'zero':
            self.conv.weight.data.zero_()
            if not add_actnorm:
                self.conv.bias.data.zero_()
        elif init_mode == 'normal':
            self.conv.weight.data.normal_(mean=0, std=0.1)
            if not add_actnorm:
                self.conv.bias.data.normal_(mean=0, std=0.1)
        else:
            raise NotImplementedError()

        if add_actnorm:
            self.actnorm = Actnorm(out_channels, 3)

        self.log_scale = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'actnorm'):
            x, _ = self.actnorm(x)
        else:
            x = x * torch.exp(self.log_scale)
        return x


class Actnorm(nn.Module):
    def __init__(self, num_chs, log_scale_factor=1):
        super(Actnorm, self).__init__()
        self.log_scale_factor = log_scale_factor
        size = [1, num_chs, 1, 1]
        bias = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        log_scale = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        self.register_parameter('bias', nn.Parameter(torch.Tensor(bias), requires_grad=True))
        self.register_parameter('log_scale', nn.Parameter(torch.Tensor(log_scale), requires_grad=True))
        self.inited = False

    def init(self, x):
        if not self.training:
            raise ValueError('In eval() mode, but ActNorm not initialized')
        with torch.no_grad():
            bias = -torch.mean(x.clone(), dim=[0, 2, 3], keepdim=True)
            var = -torch.mean((x.clone()+bias)**2, dim=[0, 2, 3], keepdim=True)
            log_scale = torch.log(1/torch.sqrt(var)+1e-6)
            self.bias.data.copy_(bias)
            self.log_scale.data.copy_(log_scale)
            self.inited = True

    def forward(self, x, log_det=0, reverse=False):
        if not self.inited:
            self.init(x)

        dims = x.shape[2] * x.shape[3]
        if not reverse:
            x += self.bias
            x *= torch.exp(self.log_scale*self.log_scale_factor)
            dlog_det = torch.sum(self.log_scale*self.log_scale_factor) * dims
            log_det = log_det + dlog_det
        else:
            x *= torch.exp(-self.log_scale*self.log_scale_factor)
            x -= self.bias
            dlog_det = -torch.sum(self.log_scale*self.log_scale_factor) * dims
            log_det = log_det + dlog_det
        return x, log_det


class Conv1x1(nn.Module):
    def __init__(self, num_chs):
        super(Conv1x1, self).__init__()
        weight = torch.randn(num_chs, num_chs)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def get_weight(self, x, reverse):
        if not reverse:
            weight = self.weight
        else:
            weight = torch.inverse(self.weight)
        dims = x.shape[2] * x.shape[3]
        dlog_det = torch.slogdet(weight)[1] * dims
        return weight, dlog_det

    def forward(self, x, log_det=None, reverse=False):
        weight, dlog_det = self.get_weight(x, reverse)
        z = F.conv2d(x, weight)
        if log_det is not None:
            if not reverse:
                log_det = log_det + dlog_det
            else:
                log_det = log_det - dlog_det
        return z, log_det


class AffineCoupling(nn.Module):
    def __init__(self, input_sz, affine_conv_chs):
        super(AffineCoupling, self).__init__()
        self.affine = nn.Sequential(
            nn.Conv2d(input_sz[0]//2, affine_conv_chs, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(affine_conv_chs, affine_conv_chs, kernel_size=1),
            nn.ReLU(inplace=True),
            Conv2d(affine_conv_chs, 2*input_sz[0]))

    def forward(self, x, log_det=0, reverse=False):
        z1, z2 = split_feature(x, 'split')
        tmp = self.affine(z1)
        shift, scale = split_feature(tmp, 'cross')
        scale = torch.sigmoid(scale+2)
        if not reverse:
            z2 += shift
            z2 *= scale
            log_det += torch.sum(torch.log(scale), dim=(1, 2, 3))
        else:
            z2 /= scale
            z2 -= shift
            log_det -= torch.sum(torch.log(scale), dim=(1, 2, 3))
        z = torch.cat((z1, z2), dim=1)
        return z, log_det


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super(SqueezeLayer, self).__init__()
        self.factor = factor

    def forward(self, x, log_det=None, reverse=False):
        if not reverse:
            x = self.squeeze2d(x, self.factor)
        else:
            x = self.unsqueeze2d(x, self.factor)
        return x, log_det

    def squeeze2d(self, x, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor != 1:
            b, c, h, w = x.shape
            assert h%factor==0 and w%factor==0, f'(h, w) = ({h}, {w})'
            x = x.reshape(b, c, h//factor, factor, w//factor, factor)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.reshape(b, c*factor*factor, h//factor, w//factor)
        return x

    def unsqueeze2d(self, x, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor != 1:
            b, c, h, w = x.shape
            assert c%(factor**2) == 0, f'c = {c}'
            x = x.reshape(b, c//(factor**2), factor, factor, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.reshape(b, c//(factor**2), h*factor, w*factor)
        return x


class Split2d(nn.Module):
    def __init__(self, num_chs):
        super(Split2d, self).__init__()
        self.conv = Conv2d(num_chs//2, num_chs)

    def split2d_prior(self, z):
        tmp = self.conv(z)
        return split_feature(tmp, 'cross')

    def forward(self, x, log_det=0, reverse=False):
        if not reverse:
            z1, z2 = split_feature(x, 'split')
            mean, log_std = self.split2d_prior(z1)
            log_det += log_likelihood(z2, mean, log_std)
            return z1, log_det
        else:
            z1 = x
            mean, log_std = self.split2d_prior(z1)
            z2 = sample(mean, log_std)
            z = torch.cat((z1, z2), dim=1)
            return z, log_det
