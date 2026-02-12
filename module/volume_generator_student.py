# student volume generator with one branch

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPStudent(torch.nn.Module):
    def __init__(self, ch_in, ch_hid, ch_out):
        super(MLPStudent, self).__init__()
        self.linear1 = torch.nn.Conv3d(ch_in, ch_hid, (1, 1, 1))
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Conv3d(ch_hid, ch_out, (1, 1, 1))

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class GeneratorStudent(torch.nn.Module):
    def __init__(self, opts):
        super(GeneratorStudent, self).__init__()
        ch_in = opts.base_channel
        ch_hid = max(4, ch_in)
        self.mapping = MLPStudent(3 * ch_in + 6, ch_hid, ch_out=ch_in)

    def forward(self, spherical_feats):
        f0, f1, f2 = spherical_feats[0], spherical_feats[1], spherical_feats[2]
        g0, g1, g2 = spherical_feats[3], spherical_feats[4], spherical_feats[5]
        concat = torch.cat([f0, f1, f2, g0, g1, g2], dim=1)
        context_feat = self.mapping(concat)
        return [context_feat, context_feat], context_feat
