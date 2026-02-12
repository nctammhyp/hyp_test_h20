import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common import *


class Conv2D(torch.nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, pad=1,
                 dilation=1, bn=True, relu=True):
        super(Conv2D, self).__init__()
        self.opts = Edict()
        self.opts.bn = bn
        self.opts.relu = relu
        self.conv = torch.nn.Conv2d(ch_in, ch_out, kernel_size,
                                    stride, padding=pad, dilation=dilation)
        if self.opts.bn:
            self.bn = torch.nn.BatchNorm2d(ch_out)

    def forward(self, x, residual=None):
        x = self.conv(x)
        if self.opts.bn:
            x = self.bn(x)
        if residual is not None:
            x += residual
        if self.opts.relu:
            x = F.relu(x)
        return x


class FeatureLayersStudent(torch.nn.Module):
    def __init__(self, CH=8, use_rgb=False, downsample_twice=False):
        super(FeatureLayersStudent, self).__init__()
        layers = []
        self.use_rgb = use_rgb
        self.downsample_twice = downsample_twice
        in_channel = 3 if use_rgb else 1

        if downsample_twice:
            layers.append(nn.Sequential(
                Conv2D(in_channel, CH, 5, 2, 2),
                Conv2D(CH, CH, 3, 2, 2, dilation=2),
            ))
        else:
            layers.append(Conv2D(in_channel, CH, 5, 2, 2))

        # Lightweight residual blocks (1/3 teacher depth)
        layers += [Conv2D(CH, CH, 3, 1, 1) for _ in range(6)]
        # Small dilated tail
        layers += [Conv2D(CH, CH, 3, 1, 2, dilation=2), Conv2D(CH, CH, 3, 1, 1, bn=False, relu=False)]

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, im):
        is_list = isinstance(im, tuple) or isinstance(im, list)
        if is_list:
            num_input = len(im)
            batch_dim = im[0].shape[0]
            im = torch.cat(im, dim=0)

        if not self.use_rgb:
            im = torch.sum(im, dim=1, keepdim=True)

        x = self.layers[0](im)
        # Residual pairs
        for i in range(1, 7, 2):
            x_ = self.layers[i](x)
            x = self.layers[i + 1](x_, residual=x)
        # Tail
        x = self.layers[7](x)
        x = self.layers[8](x)

        if is_list:
            x = torch.split(x, num_input * [batch_dim], dim=0)
        return x
