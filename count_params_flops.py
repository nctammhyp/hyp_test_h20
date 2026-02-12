import argparse
import os
import sys

import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Count params and conv FLOPs from a pruned checkpoint")
    parser.add_argument(
        "--ckpt",
        default="checkpoints/romnistereo32_v21_bs8_prune_final.pth",
        help="Path to train_prune_v2.py checkpoint",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of update iterations (default: 5)",
    )
    parser.add_argument(
        "--input_h",
        type=int,
        default=384,
        help="Input height",
    )
    parser.add_argument(
        "--input_w",
        type=int,
        default=400,
        help="Input width",
    )
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
    # 2 for mul+add
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


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        sys.exit(f"Checkpoint not found: {args.ckpt}")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        sys.exit("Checkpoint does not contain 'model' object")

    model = checkpoint["model"]
    model.eval()
    model.cpu()

    total, trainable = count_params(model)
    print(f"Checkpoint: {args.ckpt}")
    if "original_params" in checkpoint:
        print(f"original_params: {checkpoint['original_params']}")
    if "final_params" in checkpoint:
        print(f"final_params: {checkpoint['final_params']}")
    print(f"model_total_params: {total}")
    print(f"model_trainable_params: {trainable}")

    data_opts = checkpoint.get("data_opts", {})
    num_downsample = data_opts.get("num_downsample", 1)
    equirect_size = data_opts.get("equirect_size", [128, 400])
    num_invdepth = data_opts.get("num_invdepth", 48)

    h_out = equirect_size[0] // (2 ** num_downsample)
    w_out = equirect_size[1] // (2 ** num_downsample)
    d_out = num_invdepth // (2 ** num_downsample)

    img0 = torch.randn(1, 1, args.input_h, args.input_w)
    img1 = torch.randn(1, 1, args.input_h, args.input_w)
    img2 = torch.randn(1, 1, args.input_h, args.input_w)

    grid_shape = (h_out, w_out, d_out, 2)
    grid0 = torch.randn(*grid_shape)
    grid1 = torch.randn(*grid_shape)
    grid2 = torch.randn(*grid_shape)

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
        _ = model([img0, img1, img2], [grid0, grid1, grid2], iters=args.iters, test_mode=True)

    for h in hooks:
        h.remove()

    total_flops = flops["conv2d"] + flops["conv3d"]
    print(f"conv2d_flops: {flops['conv2d']}")
    print(f"conv3d_flops: {flops['conv3d']}")
    print(f"total_conv_flops: {total_flops}")
    print("note: FLOPs only include Conv2d/Conv3d; other ops are not counted")


if __name__ == "__main__":
    main()
