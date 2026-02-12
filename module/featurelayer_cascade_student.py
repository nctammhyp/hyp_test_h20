# student feature encoder with 6 conv layers

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import *
from module.featurelayer import Conv2D


class FeatureLayersStudent(torch.nn.Module):
    def __init__(self, CH=4, use_rgb=False, downsample_twice=False):
        super(FeatureLayersStudent, self).__init__()
        self.use_rgb = use_rgb
        self.downsample_twice = downsample_twice
        in_channel = 3 if use_rgb else 1

        layers = []
        if downsample_twice:
            layers.append(nn.Sequential(
                Conv2D(in_channel, CH, 5, 2, 2),
                Conv2D(CH, CH, 3, 2, 2, dilation=2)
            ))
        else:
            layers.append(Conv2D(in_channel, CH, 5, 2, 2))

        layers.append(Conv2D(CH, CH, 3, 1, 1))
        layers.append(Conv2D(CH, CH, 3, 1, 1))
        layers.append(Conv2D(CH, CH, 3, 1, 1))
        layers.append(Conv2D(CH, CH, 3, 1, 1))
        layers.append(Conv2D(CH, CH, 3, 1, 1, bn=False, relu=False))

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
        x1 = self.layers[1](x)
        x = self.layers[2](x1, residual=x)
        x2 = self.layers[3](x)
        x = self.layers[4](x2, residual=x)
        x = self.layers[5](x)

        if is_list:
            x = torch.split(x, num_input * [batch_dim], dim=0)
        return x
