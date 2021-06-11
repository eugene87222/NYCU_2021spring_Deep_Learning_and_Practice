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
            self.actnorm = Actnorm(out_channels)
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
            self.log_scale = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'actnorm'):
            x, _ = self.actnorm(x)
        else:
            x = x + self.bias
            x = x * torch.exp(self.log_scale*3)
        return x


class Actnorm(nn.Module):
    def __init__(self, num_chs):
        super(Actnorm, self).__init__()
        size = [1, num_chs, 1, 1]
        self.register_parameter('bias', nn.Parameter(torch.zeros(*size), requires_grad=True))
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(*size), requires_grad=True))
        self.inited = False

    def init(self, x):
        if not self.training:
            raise ValueError('eval() mode but not initialized')
        with torch.no_grad():
            bias = -torch.mean(x.clone(), dim=(0, 2, 3), keepdim=True)
            var = torch.mean((x.clone()+bias)**2, dim=(0, 2, 3), keepdim=True)
            log_scale = torch.log(self.scale/(torch.sqrt(var)+1e-6))
            self.bias.data.copy_(bias.data)
            self.log_scale.data.copy_(log_scale.data)
            self.inited = True

    def forward(self, x, log_det=0, reverse=False):
        if not self.inited:
            self.init(x)

        dims = x.shape[2] * x.shape[3]
        if not reverse:
            x += self.bias
            x *= torch.exp(self.log_scale)
            dlog_det = torch.sum(self.log_scale) * dims
            log_det = log_det + dlog_det
        else:
            x *= torch.exp(-self.log_scale)
            x -= self.bias
            dlog_det = -torch.sum(self.log_scale) * dims
            log_det = log_det + dlog_det
        return x, log_det


class CondActnorm(nn.Module):
    def __init__(self, input_sz, cond_sz, cond_fc_fts):
        super(CondActnorm, self).__init__()

        # self.cond_net = nn.Sequential(
        #     Linear(
        #         in_features=cond_sz,
        #         out_features=2*input_sz[0],
        #         init_mode='zero'),
        #     nn.Tanh())
        self.cond_net = nn.Sequential(
            Linear(cond_sz, cond_fc_fts, init_mode='zero'),
            nn.ReLU(inplace=True),
            Linear(cond_fc_fts, cond_fc_fts, init_mode='zero'),
            nn.ReLU(inplace=True),
            Linear(cond_fc_fts, 2*input_sz[0], init_mode='zero'),
            nn.Tanh())

    def forward(self, x, cond, log_det=0, reverse=False):
        cond_b, _, = cond.shape
        cond = self.cond_net(cond)
        cond = cond.reshape(cond_b, -1, 1, 1)

        log_scale, bias = split_feature(cond, 'split')
        dims = x.shape[2] * x.shape[3]

        if not reverse:
            x = x + bias
            x = x * torch.exp(log_scale)
            dlog_det = torch.sum(log_scale, dim=(1, 2, 3)) * dims
            log_det = log_det + dlog_det
        else:
            x = x * torch.exp(-log_scale)
            x = x - bias
            dlog_det = -torch.sum(log_scale, dim=(1, 2, 3)) * dims
            log_det = log_det + dlog_det

        return x, log_det


class CondConv1x1(nn.Module):
    def __init__(self, input_sz, cond_sz, cond_fc_fts):
        super(CondConv1x1, self).__init__()

        self.cond_net = nn.Sequential(
            Linear(cond_sz, cond_fc_fts, init_mode='zero'),
            nn.ReLU(inplace=True),
            Linear(cond_fc_fts, cond_fc_fts, init_mode='zero'),
            nn.ReLU(inplace=True),
            Linear(cond_fc_fts, input_sz[0]*input_sz[0], init_mode='normal'),
            nn.Tanh())

        # self.cond_net = nn.ConvTranspose2d(cond_sz, 1, input_sz[0])
        # nn.init.orthogonal_(self.cond_net.weight.data)

    def get_weight(self, x, cond, reverse):
        x_c = x.shape[1]
        cond_b, _ = cond.shape
        # cond = cond.reshape(cond.shape[0], cond.shape[1], 1, 1)
        cond = self.cond_net(cond)
        cond = torch.tanh(cond)
        weight = cond.reshape(cond_b, x_c, x_c)

        dims = x.shape[2] * x.shape[3]
        dlog_det = torch.slogdet(weight)[1] * dims
        if reverse:
            weight = torch.inverse(weight)
        weight = weight.reshape(cond_b, x_c, x_c, 1, 1)
        return weight, dlog_det

    def forward(self, x, cond, log_det=None, reverse=False):
        weight, dlog_det = self.get_weight(x, cond, reverse)
        x_b, x_c, x_h, x_w = x.shape
        x = x.reshape(1, x_b*x_c, x_h, x_w)
        w_b, w_c, _, w_h, w_w = weight.shape
        assert x_b==w_b and x_c==w_c, 'The input and kernel dimensions are different'
        weight = weight.reshape(w_b*w_c, w_c, w_h, w_w)

        z = F.conv2d(x, weight, groups=x_b)
        z = z.reshape(x_b, x_c, x_h, x_w)
        if log_det is not None:
            if not reverse:
                log_det = log_det + dlog_det
            else:
                log_det = log_det - dlog_det
        return z, log_det


class CondAffineCoupling(nn.Module):
    def __init__(self, input_sz, cond_sz, affine_conv_chs):
        super(CondAffineCoupling, self).__init__()

        self.input_sz = input_sz
        # self.cond_net = nn.Sequential(
        #     Linear(
        #         in_features=cond_sz,
        #         out_features=input_sz[0]*input_sz[1]*input_sz[2],
        #         init_mode='normal'),
        #     nn.ReLU(inplace=True))

        self.cond_net = nn.Sequential(
            Linear(cond_sz, 64, init_mode='zero'),
            nn.ReLU(inplace=True),
            Linear(64, 64, init_mode='zero'),
            nn.ReLU(inplace=True),
            Linear(64, input_sz[0]*input_sz[1]*input_sz[2], init_mode='zero'),
            nn.ReLU(inplace=True))

        self.affine = nn.Sequential(
            Conv2d(input_sz[0]*2, affine_conv_chs, init_mode='normal', add_actnorm=True),
            nn.ReLU(inplace=True),
            Conv2d(affine_conv_chs, affine_conv_chs, kernel_size=[1, 1], init_mode='normal', add_actnorm=True),
            nn.ReLU(inplace=True),
            Conv2d(affine_conv_chs, 2*input_sz[0]),
            nn.Tanh())

    def forward(self, x, cond, log_det=0, reverse=False):
        z1, z2 = split_feature(x, 'split')
        cond = self.cond_net(cond)
        cond = cond.reshape(cond.shape[0], self.input_sz[0], self.input_sz[1], self.input_sz[2])

        tmp = torch.cat((z1, cond), dim=1)
        tmp = self.affine(tmp)
        shift, scale = split_feature(tmp, 'cross')

        scale = torch.sigmoid(scale+2)
        if not reverse:
            z2 = z2 + shift
            z2 = z2 * scale
            log_det = log_det + torch.sum(torch.log(scale), dim=(1, 2, 3))
        else:
            z2 = z2 / scale
            z2 = z2 - shift
            log_det = log_det - torch.sum(torch.log(scale), dim=(1, 2, 3))

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
            assert h%factor==0 and w%factor==0, f'(h, w)=({h},{w}), factor={factor}'
            x = x.reshape(b, c, h//factor, factor, w//factor, factor)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.reshape(b, c*factor*factor, h//factor, w//factor)
        return x

    def unsqueeze2d(self, x, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor != 1:
            b, c, h, w = x.shape
            assert c%(factor**2) == 0, f'c={c}'
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
