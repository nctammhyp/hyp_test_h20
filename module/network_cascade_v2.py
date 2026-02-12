# network_cascade_v2.py
# Cascade Cost Volume variant with stage-aware corr levels

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from module.featurelayer import FeatureLayers, Conv2D
from module.volume_generator import Generator
from module.corr import CorrBlock1D
from module.update_cascade import UpdateBlockCascade
from utils.common import *

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def _to_edict(obj):
    if isinstance(obj, Edict):
        return obj
    if isinstance(obj, dict):
        return Edict(obj)
    return obj


class ROmniStereoCascadeV2(torch.nn.Module):
    # Cascade Cost Volume version focused on FPS with stage-aware corr levels.

    def __init__(self, varargin=None):
        super(ROmniStereoCascadeV2, self).__init__()
        opts = Edict()
        opts.use_rgb = False
        self.opts = argparse(opts, varargin)

        self.encoder = FeatureLayers(self.opts.base_channel, self.opts.use_rgb, self.opts.encoder_downsample_twice)
        context_dim = self.opts.base_channel
        hidden_dim = self.opts.base_channel * 2
        self.volume_gen = Generator(self.opts)
        self.state_conv = Conv2D(context_dim, hidden_dim, 1, pad=0, relu=False)

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        r = self.opts.corr_radius
        dx_tensor = torch.linspace(-r, r, 2 * r + 1).view(2 * r + 1, 1)
        self.register_buffer("corr_dx", dx_tensor)

        self.cascade_stages = self._build_cascade_stages()
        self.update_blocks = self._build_update_blocks()

    def _build_cascade_stages(self):
        stages = None
        if hasattr(self.opts, "cascade_stages"):
            stages = self.opts.cascade_stages
        if not stages:
            stages = [
                {"name": "coarse", "downsample": 2, "depth_stride": 4, "iters": 2},
                {"name": "mid", "downsample": 1, "depth_stride": 2, "iters": 1},
            ]

        norm = []
        for i, st in enumerate(stages):
            st = _to_edict(st)
            if isinstance(st, (list, tuple)):
                downsample, depth_stride, iters = st[:3]
                name = f"stage{i+1}"
            else:
                downsample = int(st.get("downsample", 1))
                depth_stride = int(st.get("depth_stride", 1))
                iters = int(st.get("iters", 2))
                name = st.get("name", f"stage{i+1}")

            downsample = max(1, int(downsample))
            depth_stride = max(1, int(depth_stride))
            iters = max(1, int(iters))
            norm.append(Edict({
                "name": name,
                "downsample": downsample,
                "depth_stride": depth_stride,
                "iters": iters,
            }))

        require_base = bool(getattr(self.opts, "cascade_require_base_stage", True))
        if require_base and norm[-1].downsample != 1:
            norm.append(Edict({
                "name": "base",
                "downsample": 1,
                "depth_stride": 1,
                "iters": 1,
            }))

        return norm

    def _build_update_blocks(self):
        blocks = nn.ModuleDict()
        for levels in range(1, int(self.opts.corr_levels) + 1):
            blocks[str(levels)] = UpdateBlockCascade(
                self.opts,
                hidden_dim=self.hidden_dim,
                input_dim=self.context_dim,
                corr_levels=levels,
                corr_radius=self.opts.corr_radius,
            )
        return blocks

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _make_stage_grids(self, grids, hw_stride, d_stride):
        stage_grids = []
        for grid in grids:
            g = grid
            if hw_stride > 1:
                g = g[::hw_stride, ::hw_stride, :, :]
            if d_stride > 1:
                g = g[:, :, ::d_stride, :]
            stage_grids.append(g)
        return stage_grids

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

    def _scale_invdepth(self, invdepth_idx, d_prev, d_new):
        if d_prev <= 1:
            return torch.zeros_like(invdepth_idx)
        scale = float(d_new - 1) / float(d_prev - 1)
        return invdepth_idx * scale

    def _resize_invdepth(self, invdepth_idx, h_new, w_new):
        if invdepth_idx.shape[-2] == h_new and invdepth_idx.shape[-1] == w_new:
            return invdepth_idx
        return F.interpolate(invdepth_idx, size=(h_new, w_new), mode="bilinear", align_corners=True)

    def forward(self, imgs, grids, iters=None, test_mode=False):
        with autocast(enabled=self.opts.mixed_precision):
            fisheye_feats = self.encoder(imgs)

        fisheye_feats = [feat.float() for feat in fisheye_feats]

        invdepth_idx_predictions = []
        prev_invdepth = None
        prev_d = None
        prev_hw = None

        for stage_idx, stage in enumerate(self.cascade_stages):
            stage_grids = self._make_stage_grids(grids, stage.downsample, stage.depth_stride)
            spherical_feats = self.spherical_sweep(fisheye_feats, stage_grids)

            with autocast(enabled=self.opts.mixed_precision):
                match_feat_volume_list, context_feat_volume = self.volume_gen(spherical_feats)

            context_feat = context_feat_volume[..., 0]

            with autocast(enabled=self.opts.mixed_precision):
                inp = torch.relu(context_feat)
                net = torch.tanh(self.state_conv(context_feat))

            match_feat_volume_list = [feat.float() for feat in match_feat_volume_list]
            d_stage = context_feat_volume.shape[-1]
            if d_stage <= 1:
                max_levels = 1
            else:
                max_levels = int(math.floor(math.log2(d_stage))) + 1
            corr_levels = min(int(self.opts.corr_levels), max_levels)
            corr_fn = CorrBlock1D(*match_feat_volume_list,
                                  dx_buffer=self.corr_dx,
                                  radius=self.opts.corr_radius,
                                  num_levels=corr_levels)

            if prev_invdepth is None:
                invdepth_idx = torch.zeros_like(context_feat_volume[:, :1, ..., 0])
            else:
                h_cur, w_cur = context_feat_volume.shape[-3], context_feat_volume.shape[-2]
                invdepth_idx = prev_invdepth
                if prev_hw is not None:
                    invdepth_idx = self._resize_invdepth(invdepth_idx, h_cur, w_cur)
                invdepth_idx = self._scale_invdepth(invdepth_idx, prev_d, d_stage)

            update_block = self.update_blocks[str(corr_levels)]
            stage_iters = stage.iters if iters is None else iters
            for itr in range(stage_iters):
                invdepth_idx = invdepth_idx.detach()
                corr_feat = corr_fn(invdepth_idx)
                if itr > 0:
                    context_feat = self.volume_sample(context_feat_volume, invdepth_idx)
                    inp = torch.relu(context_feat)
                with autocast(enabled=self.opts.mixed_precision):
                    no_upsample = (stage_idx < len(self.cascade_stages) - 1) or (test_mode and itr < stage_iters - 1)
                    net, delta_invdepth_idx, up_mask = update_block(
                        net, inp, corr_feat, invdepth_idx, no_upsample=no_upsample
                    )
                invdepth_idx = invdepth_idx + delta_invdepth_idx

                if (stage_idx == len(self.cascade_stages) - 1) and (up_mask is not None):
                    invdepth_idx_up = self.upsample_invdepth_idx(invdepth_idx, up_mask)
                    invdepth_idx_predictions.append(invdepth_idx_up)

            prev_invdepth = invdepth_idx
            prev_d = d_stage
            prev_hw = (context_feat_volume.shape[-3], context_feat_volume.shape[-2])

        if test_mode:
            if invdepth_idx_predictions:
                return torch.clamp(invdepth_idx_predictions[-1], 0, self.opts.num_invdepth - 1)
            return torch.clamp(prev_invdepth, 0, self.opts.num_invdepth - 1)

        return invdepth_idx_predictions
