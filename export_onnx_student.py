import argparse
import os

import onnx
import torch
import torch.nn as nn
from easydict import EasyDict as Edict

from module.network_student import ROmniStereoStudent


class StudentONNX(nn.Module):
    def __init__(self, model, iters):
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, img0, img1, img2, grid0, grid1, grid2):
        imgs = [img0, img1, img2]
        grids = [grid0, grid1, grid2]
        return self.model(imgs, grids, iters=self.iters, test_mode=True)


def main():
    parser = argparse.ArgumentParser(description="Export Student Model to ONNX")
    parser.add_argument("--output_path", type=str, default="checkpoints/onnx/romnistereo_student.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--iters", type=int, default=2)
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
    model.cpu()

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

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    onnx_model = StudentONNX(model, args.iters)
    torch.onnx.export(
        onnx_model,
        (img0, img1, img2, grid0, grid1, grid2),
        args.output_path,
        input_names=["img0", "img1", "img2", "grid0", "grid1", "grid2"],
        output_names=["depth_map"],
        opset_version=args.opset,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
    )

    model_onnx = onnx.load(args.output_path)
    onnx.checker.check_model(model_onnx)
    print(f"Exported student ONNX: {args.output_path}")


if __name__ == "__main__":
    main()
