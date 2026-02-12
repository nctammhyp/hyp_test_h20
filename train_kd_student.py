from __future__ import print_function, division

import os
import sys
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from torch.cuda.amp import GradScaler
except Exception:
    class GradScaler:
        def __init__(self, enabled=False):
            self.enabled = enabled
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

from dataset import Dataset, MultiDataset
from module.network import ROmniStereo
from module.loss_functions import sequence_loss
from utils.common import *

torch.backends.cudnn.benchmark = True
torch.backends.cuda.benchmark = True


def build_opts(args):
    opts = Edict()

    opts.name = args.name
    opts.model_dir = os.path.join("./checkpoints", args.name)
    opts.runs_dir = os.path.join("./runs", args.name)

    opts.snapshot_path = args.restore_ckpt
    opts.pretrain_path = args.pretrain_ckpt

    opts.dbname = args.dbname
    opts.db_root = args.db_root

    opts.data_opts = Edict()
    opts.data_opts.phi_deg = args.phi_deg
    opts.data_opts.num_invdepth = args.num_invdepth
    opts.data_opts.equirect_size = args.equirect_size
    opts.data_opts.num_downsample = args.num_downsample
    opts.data_opts.use_rgb = args.use_rgb

    opts.net_opts = Edict()
    opts.net_opts.base_channel = args.base_channel
    opts.net_opts.num_invdepth = opts.data_opts.num_invdepth
    opts.net_opts.use_rgb = opts.data_opts.use_rgb
    opts.net_opts.encoder_downsample_twice = args.encoder_downsample_twice
    opts.net_opts.num_downsample = args.num_downsample
    opts.net_opts.corr_levels = args.corr_levels
    opts.net_opts.corr_radius = args.corr_radius
    opts.net_opts.mixed_precision = args.mixed_precision
    opts.net_opts.fix_bn = args.fix_bn

    opts.total_epochs = args.total_epochs
    opts.batch_size = args.batch_size
    opts.train_iters = args.train_iters
    opts.valid_iters = args.valid_iters
    opts.lr = args.lr
    opts.wdecay = args.wdecay

    opts.kd_weight = args.kd_weight
    opts.sup_weight = args.sup_weight
    opts.teacher_iters = args.teacher_iters

    return opts


def fetch_optimizer(model, lr, wdecay):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wdecay, eps=1e-8)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_teacher(ckpt_path, fallback_net_opts):
    if not ckpt_path or not os.path.exists(ckpt_path):
        sys.exit(f"Teacher checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        teacher = ckpt["model"]
        if isinstance(teacher, nn.DataParallel):
            teacher = teacher.module
    else:
        net_opts = ckpt.get("net_opts", fallback_net_opts)
        teacher = ROmniStereo(net_opts)
        if "net_state_dict" in ckpt:
            teacher.load_state_dict(ckpt["net_state_dict"], strict=False)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def train(opts, load_state):
    if len(opts.dbname) > 1:
        data = MultiDataset(opts.dbname, opts.data_opts, db_root=opts.db_root)
    else:
        data = Dataset(opts.dbname[0], opts.data_opts, db_root=opts.db_root)

    dbloader = torch.utils.data.DataLoader(
        data,
        batch_size=opts.batch_size,
        # pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    student = nn.DataParallel(ROmniStereo(opts.net_opts)).cuda()
    if opts.net_opts.fix_bn:
        student.module.freeze_bn()

    teacher = load_teacher(opts.teacher_ckpt, opts.net_opts)
    teacher = nn.DataParallel(teacher).cuda()

    LOG_INFO("Student Parameter Count: %d" % count_parameters(student))

    optimizer = fetch_optimizer(student, opts.lr, opts.wdecay)
    scaler = GradScaler(enabled=opts.net_opts.mixed_precision)

    start_epoch = 0
    if load_state:
        if opts.snapshot_path and osp.exists(opts.snapshot_path):
            snapshot = torch.load(opts.snapshot_path, weights_only=False)
            if "net_state_dict" in snapshot:
                student.load_state_dict(snapshot["net_state_dict"], strict=False)
                LOG_INFO("checkpoint %s is loaded" % (opts.snapshot_path))
            if "epoch" in snapshot:
                start_epoch = snapshot["epoch"] + 1
            if "optimizer" in snapshot:
                optimizer.load_state_dict(snapshot["optimizer"])
        elif opts.pretrain_path is None:
            sys.exit("%s do not exsits" % (opts.snapshot_path))

        if opts.pretrain_path and osp.exists(opts.pretrain_path):
            snapshot = torch.load(opts.pretrain_path, weights_only=False)
            if "net_state_dict" in snapshot:
                student.load_state_dict(snapshot["net_state_dict"], strict=False)
                LOG_INFO("checkpoint %s is loaded" % (opts.pretrain_path))
        elif opts.snapshot_path is None:
            sys.exit("%s do not exsits" % (opts.snapshot_path))

    grids = [torch.tensor(grid, requires_grad=False).cuda() for grid in data.grids]

    os.makedirs(opts.model_dir, exist_ok=True)
    os.makedirs(opts.runs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=opts.runs_dir)

    total_iters = len(data) * start_epoch // opts.batch_size

    for epoch in range(start_epoch, opts.total_epochs):
        student.train()
        train_loss = 0.0
        LOG_INFO("\nEpoch: %d" % epoch)
        pbar = tqdm(dbloader, total=len(dbloader), desc=f"Epoch {epoch}")

        for step, data_blob in enumerate(pbar):
            imgs, gt, valid, raw_imgs = data_blob
            imgs = [img.cuda() for img in imgs]
            valid = valid.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()

            student_preds = student(imgs, grids, opts.train_iters)
            sup_loss = sequence_loss(student_preds, gt.unsqueeze(1), valid.unsqueeze(1))

            with torch.no_grad():
                teacher_pred = teacher(imgs, grids, opts.teacher_iters, test_mode=True)

            student_last = student_preds[-1]
            kd_loss = (student_last - teacher_pred).abs().mean()

            loss = opts.sup_weight * sup_loss + opts.kd_weight * kd_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.detach().item()
            epoch_loss = train_loss / (step + 1)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sup": f"{sup_loss.item():.4f}",
                "kd": f"{kd_loss.item():.4f}",
                "avg": f"{epoch_loss:.4f}",
            })

            total_iters += 1

        writer.add_scalar("train/epoch_loss", epoch_loss, total_iters)

        # save student checkpoint
        savefilename = opts.model_dir + "/%s_e%d.pth" % (opts.name, epoch)
        torch.save({
            "net_state_dict": student.state_dict(),
            "net_opts": opts.net_opts,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "epoch_loss": epoch_loss,
        }, savefilename)


def main():
    parser = ArgumentParser(description="Knowledge Distillation Training for ROmniStereo student")

    parser.add_argument("--name", default="ROmniStereo_KD", help="experiment name")
    parser.add_argument("--restore_ckpt", help="restore checkpoint")
    parser.add_argument("--pretrain_ckpt", help="pretrained checkpoint for finetuning")
    parser.add_argument("--teacher_ckpt", required=True, help="teacher checkpoint path")

    parser.add_argument(
        "--db_root",
        default="/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/hyp_data_01_trainable/",
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        "--dbname",
        nargs="+",
        default=["omnithings"],
        type=str,
        choices=["omnithings", "omnihouse", "sunny", "cloudy", "sunset"],
        help="databases to train",
    )

    parser.add_argument("--phi_deg", type=float, default=45.0, help="phi_deg")
    parser.add_argument("--num_invdepth", type=int, default=32, help="number of disparity")
    parser.add_argument("--equirect_size", type=int, nargs="+", default=[128, 400], help="out ERP size")
    parser.add_argument("--use_rgb", action="store_true", help="use 3-channel rgb input")

    parser.add_argument("--base_channel", type=int, default=16, help="student base channel")
    parser.add_argument("--encoder_downsample_twice", action="store_true", help="downsample twice")
    parser.add_argument("--num_downsample", type=int, default=1)
    parser.add_argument("--corr_levels", type=int, default=4)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--fix_bn", action="store_true")

    parser.add_argument("--total_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_iters", type=int, default=3)
    parser.add_argument("--valid_iters", type=int, default=3)
    parser.add_argument("--teacher_iters", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--wdecay", type=float, default=0.00001)

    parser.add_argument("--kd_weight", type=float, default=0.5, help="KD loss weight")
    parser.add_argument("--sup_weight", type=float, default=1.0, help="Supervised loss weight")

    args = parser.parse_args()

    opts = build_opts(args)
    opts.teacher_ckpt = args.teacher_ckpt

    load_state = opts.snapshot_path is not None or opts.pretrain_path is not None
    train(opts, load_state)


if __name__ == "__main__":
    main()
