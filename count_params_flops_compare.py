import argparse
import os
import sys

import torch
import torch.nn as nn

from utils.common import Edict, argparse as edict_argparse
from module.network import ROmniStereo
from module.network_cascade_v2 import ROmniStereoCascadeV2


def parse_args():
    parser = argparse.ArgumentParser(description="Compare params/FLOPs for normal vs cascade")

    # common data/model options
    parser.add_argument("--input_h", type=int, default=384, help="Input fisheye height")
    parser.add_argument("--input_w", type=int, default=400, help="Input fisheye width")
    parser.add_argument("--equirect_size", type=int, nargs="+", default=[160, 640], help="ERP size (H W)")
    parser.add_argument("--num_invdepth", type=int, default=32, help="Number of invdepth bins")
    parser.add_argument("--num_downsample", type=int, default=1, help="Downsample factor in network")
    parser.add_argument("--base_channel", type=int, default=16, help="Base channel")
    parser.add_argument("--corr_levels", type=int, default=4, help="Correlation pyramid levels")
    parser.add_argument("--corr_radius", type=int, default=4, help="Correlation radius")
    parser.add_argument("--encoder_downsample_twice", action="store_true")
    parser.add_argument("--use_rgb", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")

    # normal model iters
    parser.add_argument("--iters", type=int, default=2, help="Update iterations for normal model")

    # cascade options
    parser.add_argument("--cascade_s1_downsample", type=int, default=2)
    parser.add_argument("--cascade_s1_depth_stride", type=int, default=4)
    parser.add_argument("--cascade_s1_iters", type=int, default=2)
    parser.add_argument("--cascade_s2_downsample", type=int, default=1)
    parser.add_argument("--cascade_s2_depth_stride", type=int, default=1)
    parser.add_argument("--cascade_s2_iters", type=int, default=1)
    parser.add_argument("--cascade_require_base_stage", action="store_true")

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for forward")
    return parser.parse_args()


def count_params(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable


def conv2d_flops(module, inp, out):
    x = inp[0]
    if x is None:
        return 0
    batch = x.shape[0]
    out_h, out_w = out.shape[-2], out.shape[-1]
    kernel_h, kernel_w = module.kernel_size
    in_ch = module.in_channels
    out_ch = module.out_channels
    groups = module.groups
    return int(batch * out_h * out_w * out_ch * (in_ch // groups) * kernel_h * kernel_w * 2)


def conv3d_flops(module, inp, out):
    x = inp[0]
    if x is None:
        return 0
    batch = x.shape[0]
    out_d, out_h, out_w = out.shape[-3], out.shape[-2], out.shape[-1]
    k_d, k_h, k_w = module.kernel_size
    in_ch = module.in_channels
    out_ch = module.out_channels
    groups = module.groups
    return int(batch * out_d * out_h * out_w * out_ch * (in_ch // groups) * k_d * k_h * k_w * 2)


def make_opts(args):
    opts = Edict()
    opts.use_rgb = args.use_rgb
    opts.base_channel = args.base_channel
    opts.num_invdepth = args.num_invdepth
    opts.encoder_downsample_twice = args.encoder_downsample_twice
    opts.num_downsample = args.num_downsample
    opts.corr_levels = args.corr_levels
    opts.corr_radius = args.corr_radius
    opts.mixed_precision = args.mixed_precision
    opts.fix_bn = False
    return opts


def make_cascade_opts(args):
    opts = make_opts(args)
    opts.cascade_require_base_stage = args.cascade_require_base_stage
    opts.cascade_stages = [
        {"name": "coarse", "downsample": args.cascade_s1_downsample, "depth_stride": args.cascade_s1_depth_stride, "iters": args.cascade_s1_iters},
        {"name": "mid", "downsample": args.cascade_s2_downsample, "depth_stride": args.cascade_s2_depth_stride, "iters": args.cascade_s2_iters},
    ]
    return opts


def build_grids(h_out, w_out, d_out, device):
    grid_shape = (h_out, w_out, d_out, 2)
    grid0 = torch.randn(*grid_shape, device=device)
    grid1 = torch.randn(*grid_shape, device=device)
    grid2 = torch.randn(*grid_shape, device=device)
    return [grid0, grid1, grid2]


def run_and_count(model, imgs, grids, iters, test_mode=True):
    flops = {"conv2d": 0, "conv3d": 0}
    hooks = []

    def hook_fn(module, inp, out):
        if isinstance(module, nn.Conv2d):
            flops["conv2d"] += conv2d_flops(module, inp, out)
        elif isinstance(module, nn.Conv3d):
            flops["conv3d"] += conv3d_flops(module, inp, out)

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(imgs, grids, iters=iters, test_mode=test_mode)

    for h in hooks:
        h.remove()

    total_flops = flops["conv2d"] + flops["conv3d"]
    return flops["conv2d"], flops["conv3d"], total_flops


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    h_out = args.equirect_size[0] // (2 ** args.num_downsample)
    w_out = args.equirect_size[1] // (2 ** args.num_downsample)
    d_out = args.num_invdepth // (2 ** args.num_downsample)

    in_ch = 3 if args.use_rgb else 1
    img0 = torch.randn(1, in_ch, args.input_h, args.input_w, device=device)
    img1 = torch.randn(1, in_ch, args.input_h, args.input_w, device=device)
    img2 = torch.randn(1, in_ch, args.input_h, args.input_w, device=device)
    imgs = [img0, img1, img2]

    grids = build_grids(h_out, w_out, d_out, device)

    # Normal model
    normal_opts = make_opts(args)
    normal_model = ROmniStereo(normal_opts).to(device)
    normal_model.eval()
    total_params, trainable_params = count_params(normal_model)
    n_conv2d, n_conv3d, n_total = run_and_count(normal_model, imgs, grids, iters=args.iters, test_mode=True)

    # Cascade model
    cascade_opts = make_cascade_opts(args)
    cascade_model = ROmniStereoCascadeV2(cascade_opts).to(device)
    cascade_model.eval()
    c_total_params, c_trainable_params = count_params(cascade_model)
    c_conv2d, c_conv3d, c_total = run_and_count(cascade_model, imgs, grids, iters=None, test_mode=True)

    print("Normal model")
    print(f"  params_total: {total_params}")
    print(f"  params_trainable: {trainable_params}")
    print(f"  conv2d_flops: {n_conv2d}")
    print(f"  conv3d_flops: {n_conv3d}")
    print(f"  total_conv_flops: {n_total}")
    print(f"  note: iters={args.iters}")

    print("Cascade model")
    print(f"  params_total: {c_total_params}")
    print(f"  params_trainable: {c_trainable_params}")
    print(f"  conv2d_flops: {c_conv2d}")
    print(f"  conv3d_flops: {c_conv3d}")
    print(f"  total_conv_flops: {c_total}")
    print("  note: iters use cascade stage settings")


if __name__ == "__main__":
    main()
