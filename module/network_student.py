import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from module.featurelayer_student import FeatureLayersStudent, Conv2D
from module.volume_generator import Generator
from module.corr import CorrBlock1D
from module.update_student import UpdateBlockStudent
from utils.common import *

try:
    autocast = torch.cuda.amp.autocast
except Exception:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class ROmniStereoStudent(torch.nn.Module):
    def __init__(self, varargin=None):
        super(ROmniStereoStudent, self).__init__()
        opts = Edict()
        opts.use_rgb = False
        opts.base_channel = 8
        opts.encoder_downsample_twice = False
        opts.num_downsample = 1
        opts.num_invdepth = 16
        opts.corr_levels = 2
        opts.corr_radius = 2
        opts.mixed_precision = False
        self.opts = argparse(opts, varargin)

        self.encoder = FeatureLayersStudent(self.opts.base_channel, self.opts.use_rgb, self.opts.encoder_downsample_twice)
        context_dim = self.opts.base_channel
        hidden_dim = self.opts.base_channel * 2

        self.volume_gen = Generator(self.opts)
        self.state_conv = Conv2D(context_dim, hidden_dim, 1, pad=0, relu=False)
        self.update_block = UpdateBlockStudent(self.opts, hidden_dim=hidden_dim, input_dim=context_dim)

        r = self.opts.corr_radius
        dx_tensor = torch.linspace(-r, r, 2 * r + 1).view(2 * r + 1, 1)
        self.register_buffer("corr_dx", dx_tensor)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def spherical_sweep(self, fisheye_feats, grids):
        bs = fisheye_feats[0].shape[0]
        sph_feats = []

        for feat, grid in zip(fisheye_feats, grids):
            h_out, w_out, d_out, _ = grid.shape
            sampled_slices = []
            for d in range(d_out):
                grid_slice = grid[:, :, d, :]
                grid_slice_b = grid_slice.unsqueeze(0).repeat(bs, 1, 1, 1)
                sample = F.grid_sample(feat, grid_slice_b, align_corners=True, mode="bilinear")
                sampled_slices.append(sample)
            sampled_volume = torch.stack(sampled_slices, dim=-1)
            sph_feats.append(sampled_volume)

        for grid in grids:
            g_emb = grid.permute(3, 0, 1, 2).unsqueeze(0).repeat(bs, 1, 1, 1, 1)
            sph_feats.append(g_emb)

        return sph_feats

    def upsample_invdepth_idx(self, invdepth, mask):
        bs, ch, h, w = invdepth.shape
        factor = 2 ** self.opts.num_downsample
        mask = mask.view(bs, 1, 9, factor, factor, h, w)
        mask = torch.softmax(mask, dim=2)

        up_invdepth = F.unfold(factor * invdepth, [3, 3], padding=1)
        up_invdepth = up_invdepth.view(bs, ch, 9, 1, 1, h, w)

        up_invdepth = torch.sum(mask * up_invdepth, dim=2)
        up_invdepth = up_invdepth.permute(0, 1, 4, 2, 5, 3)
        return up_invdepth.reshape(bs, ch, factor * h, factor * w)

    def volume_sample(self, feat_volume, invdepth_idx):
        bs, ch, h, w, n_invd = feat_volume.shape
        feat_4d = feat_volume.permute(0, 1, 2, 4, 3).reshape(bs * ch * h, 1, n_invd, w)

        grid_x = torch.linspace(-1, 1, w, device=invdepth_idx.device).view(1, 1, w, 1)
        grid_x = grid_x.expand(bs * ch * h, 1, w, 1)
        norm_idx = (invdepth_idx.permute(0, 2, 3, 1).reshape(bs * h, 1, w, 1) * 2 / (n_invd - 1)) - 1
        grid_y = norm_idx.repeat_interleave(ch, dim=0)
        grid = torch.cat([grid_x, grid_y], dim=-1)
        sampled = F.grid_sample(feat_4d, grid, align_corners=True, mode="bilinear")
        return sampled.view(bs, ch, h, w)

    def forward(self, imgs, grids, iters=2, test_mode=False, dump_inputs=False):
        if dump_inputs:
            return {
                "imgs": [im.detach().cpu() for im in imgs],
                "grids": [g.detach().cpu() for g in grids],
            }

        with autocast(enabled=self.opts.mixed_precision):
            fisheye_feats = self.encoder(imgs)

        fisheye_feats = [feat.float() for feat in fisheye_feats]
        spherical_feats = self.spherical_sweep(fisheye_feats, grids)

        with autocast(enabled=self.opts.mixed_precision):
            match_feat_volume_list, context_feat_volume = self.volume_gen(spherical_feats)

        context_feat = context_feat_volume[..., 0]

        with autocast(enabled=self.opts.mixed_precision):
            inp = torch.relu(context_feat)
            net = torch.tanh(self.state_conv(context_feat))

        match_feat_volume_list = [feat.float() for feat in match_feat_volume_list]

        corr_fn = CorrBlock1D(
            *match_feat_volume_list,
            dx_buffer=self.corr_dx,
            radius=self.opts.corr_radius,
            num_levels=self.opts.corr_levels,
        )

        invdepth_idx = torch.zeros_like(context_feat_volume[:, :1, ..., 0])
        invdepth_idx_predictions = []

        for itr in range(iters):
            invdepth_idx = invdepth_idx.detach()
            corr_feat = corr_fn(invdepth_idx)
            if itr > 0:
                context_feat = self.volume_sample(context_feat_volume, invdepth_idx)
                inp = torch.relu(context_feat)
            with autocast(enabled=self.opts.mixed_precision):
                net, delta_invdepth_idx, up_mask = self.update_block(
                    net,
                    inp,
                    corr_feat,
                    invdepth_idx,
                    no_upsample=(test_mode and itr < iters - 1),
                )
            invdepth_idx = invdepth_idx + delta_invdepth_idx

            if up_mask is not None:
                invdepth_idx_up = self.upsample_invdepth_idx(invdepth_idx, up_mask)
                invdepth_idx_predictions.append(invdepth_idx_up)

        if test_mode:
            return torch.clamp(invdepth_idx_predictions[-1], 0, self.opts.num_invdepth - 1)

        return invdepth_idx_predictions
