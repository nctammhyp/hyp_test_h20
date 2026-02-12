import argparse
import os

import numpy as np
import torch
from easydict import EasyDict as Edict

from module.network_student import ROmniStereoStudent


def main():
    parser = argparse.ArgumentParser(description="Dump student inputs for testing")
    parser.add_argument("--out", default="student_inputs.npz", help="Output .npz path")
    parser.add_argument("--input_h", type=int, default=320)
    parser.add_argument("--input_w", type=int, default=320)
    parser.add_argument("--equirect_h", type=int, default=128)
    parser.add_argument("--equirect_w", type=int, default=320)
    parser.add_argument("--num_invdepth", type=int, default=16)
    args = parser.parse_args()

    opts = Edict()
    opts.use_rgb = False
    opts.base_channel = 8
    opts.encoder_downsample_twice = False
    opts.num_downsample = 1
    opts.num_invdepth = args.num_invdepth
    opts.corr_levels = 2
    opts.corr_radius = 2
    opts.mixed_precision = False

    model = ROmniStereoStudent(opts)
    model.eval()

    img0 = torch.randn(1, 1, args.input_h, args.input_w)
    img1 = torch.randn(1, 1, args.input_h, args.input_w)
    img2 = torch.randn(1, 1, args.input_h, args.input_w)

    h_out = args.equirect_h // (2 ** opts.num_downsample)
    w_out = args.equirect_w // (2 ** opts.num_downsample)
    d_out = args.num_invdepth // (2 ** opts.num_downsample)
    grid_shape = (h_out, w_out, d_out, 2)
    grid0 = torch.randn(*grid_shape)
    grid1 = torch.randn(*grid_shape)
    grid2 = torch.randn(*grid_shape)

    dump = model(
        [img0, img1, img2],
        [grid0, grid1, grid2],
        iters=2,
        test_mode=True,
        dump_inputs=True,
    )

    np.savez(
        args.out,
        img0=dump["imgs"][0].numpy(),
        img1=dump["imgs"][1].numpy(),
        img2=dump["imgs"][2].numpy(),
        grid0=dump["grids"][0].numpy(),
        grid1=dump["grids"][1].numpy(),
        grid2=dump["grids"][2].numpy(),
    )

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
