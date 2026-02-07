from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset, MultiDataset
from utils.common import *
from utils.image import *
from module.network import ROmniStereo
from module.loss_functions import sequence_loss

import torch_pruning as tp
import copy

try:
    from torch.cuda.amp import GradScaler, autocast
except:
    class GradScaler:
        def __init__(self, enabled=False): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass

# ==========================================
# 1. Khai bao ArgumentParser
# ==========================================
parser = ArgumentParser(description='Iterative Pruning with Best Checkpoint Recovery (V2 - Save Full Model)')

parser.add_argument('--name', default='ROmniPruned_v2', help="name of your experiment")
parser.add_argument('--restore_ckpt', default='/home/sw-tamnguyen/Desktop/depth_project/hyp_test_h20/checkpoints/romnistereo32_v20_bs8/romnistereo32_v20_bs8_e46.pth')
parser.add_argument('--db_root', default='/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/hyp_data_01_trainable/', type=str, help='path to dataset')

parser.add_argument('--dbname', nargs='+', default=['omnithings'], type=str)
parser.add_argument('--phi_deg', type=float, default=45.0)
parser.add_argument('--num_invdepth', type=int, default=48)
parser.add_argument('--equirect_size', type=int, nargs='+', default=[128, 400])
parser.add_argument('--use_rgb', action='store_true')
parser.add_argument('--base_channel', type=int, default=8)
parser.add_argument('--encoder_downsample_twice', action='store_true')
parser.add_argument('--num_downsample', type=int, default=1)
parser.add_argument('--corr_levels', type=int, default=4)
parser.add_argument('--corr_radius', type=int, default=4)
parser.add_argument('--mixed_precision', action='store_true')
parser.add_argument('--fix_bn', action='store_true')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--train_iters', type=int, default=5)
parser.add_argument('--valid_iters', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wdecay', type=float, default=.00001)

# PRUNING PARAMS
parser.add_argument('--pruning_ratio', type=float, default=1, help="Tong ti le muon cat (vd 0.5 la 50%)")
parser.add_argument('--pruning_steps', type=int, default=10, help="So lan thuc hien cat tia")
parser.add_argument('--fine_tune_epochs', type=int, default=5, help="So epoch toi da de phuc hoi moi buoc")

args = parser.parse_args()

opts = Edict()
opts.name = args.name
opts.model_dir = os.path.join('./checkpoints', args.name)
opts.runs_dir = os.path.join('./runs', args.name)
opts.snapshot_path = args.restore_ckpt
opts.data_opts = Edict({'phi_deg': args.phi_deg, 'num_invdepth': args.num_invdepth, 'equirect_size': args.equirect_size, 'num_downsample': args.num_downsample, 'use_rgb': args.use_rgb})
opts.net_opts = Edict({'base_channel': args.base_channel, 'num_invdepth': args.num_invdepth, 'use_rgb': args.use_rgb, 'encoder_downsample_twice': args.encoder_downsample_twice, 'num_downsample': args.num_downsample, 'corr_levels': args.corr_levels, 'corr_radius': args.corr_radius, 'mixed_precision': args.mixed_precision, 'fix_bn': args.fix_bn})

# ==========================================
# 2. Ham Validation (Tinh RMS)
# ==========================================
def validate(model, dataset, grids):
    model.eval()
    eval_list = dataset.opts.test_idx
    errors = np.zeros((len(eval_list), 5))
    
    with torch.no_grad():
        for d in range(len(eval_list)):
            fidx = eval_list[d]
            imgs, gt, valid, _ = dataset.loadSample(fidx)
            imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]
            invdepth_idx = model(imgs, grids, args.valid_iters, test_mode=True)
            invdepth_idx = toNumpy(invdepth_idx[0, 0])
            errors[d, :] = dataset.evalError(invdepth_idx, gt, valid)
            
    mean_errors = errors.mean(axis=0) # [>1, >3, >5, MAE, RMS]
    return mean_errors

# ==========================================
# 3. Ham dem so tham so cua model
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 4. Main Logic
# ==========================================
def main():
    if len(args.dbname) > 1:
        data = MultiDataset(args.dbname, opts.data_opts, db_root=args.db_root)
    else:
        data = Dataset(args.dbname[0], opts.data_opts, db_root=args.db_root)
    
    dbloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    grids = [torch.tensor(grid).cuda() for grid in data.grids]

    model = ROmniStereo(opts.net_opts).cuda()
    
    # In so tham so truoc khi prune
    original_params = count_parameters(model)
    LOG_INFO(f"Original model parameters: {original_params:,}")
    
    if opts.snapshot_path:
        checkpoint = torch.load(opts.snapshot_path, weights_only=False)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net_state_dict'].items()}
        model.load_state_dict(state_dict)
        LOG_INFO(f"Loaded: {opts.snapshot_path}")

    example_imgs = [torch.randn(1, 1, 384, 400).cuda() for _ in range(3)]
    
    # Pruner config
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels in [1, (2**args.num_downsample)**2 * 9]:
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=(example_imgs, grids),
        importance=tp.importance.MagnitudeImportance(p=2),
        iterative_steps=args.pruning_steps,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=4,
    )

    scaler = GradScaler(enabled=args.mixed_precision)
    writer = SummaryWriter(log_dir=opts.runs_dir)

    if not os.path.exists(opts.model_dir): 
        os.makedirs(opts.model_dir)

    # LOOP CAT TIA (OUTER LOOP)
    for step in range(args.pruning_steps):
        LOG_INFO(f"\n{'='*20}\nPRUNING STEP {step+1}/{args.pruning_steps}\n{'='*20}")
        
        # 1. Cat tia
        pruner.step()
        
        # In so tham so sau moi buoc prune
        current_params = count_parameters(model)
        reduction = (1 - current_params / original_params) * 100
        LOG_INFO(f"Parameters after step {step+1}: {current_params:,} (reduced {reduction:.1f}%)")
        
        # Reset optimizer sau moi lan cat tia cau truc
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        
        best_rms_in_step = float('inf')
        temp_best_ckpt = os.path.join(opts.model_dir, f"tmp_step_{step}_best.pth")

        # LOOP FINE-TUNE (INNER LOOP)
        for ft_epoch in range(args.fine_tune_epochs):
            model.train()
            pbar = tqdm(dbloader, desc=f"Step {step+1} Fine-tune Ep {ft_epoch}")
            
            for imgs_b, gt_b, valid_b, _ in pbar:
                imgs_b = [img.cuda() for img in imgs_b]
                gt_b, valid_b = gt_b.cuda(), valid_b.cuda()
                
                optimizer.zero_grad()
                preds = model(imgs_b, grids, args.train_iters)
                loss = sequence_loss(preds, gt_b.unsqueeze(1), valid_b.unsqueeze(1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # --- VALIDATION SAU MOI EPOCH ---
            metrics = validate(model, data, grids)
            current_rms = metrics[4]
            LOG_INFO(f"Epoch {ft_epoch} Result -> RMS: {current_rms:.4f} (Best so far: {best_rms_in_step:.4f})")
            
            # Neu tot hon thi luu checkpoint tam thoi (chi luu state_dict cho temp)
            if current_rms < best_rms_in_step:
                best_rms_in_step = current_rms
                torch.save(model.state_dict(), temp_best_ckpt)
                LOG_INFO(f"New best epoch found! Saved to {temp_best_ckpt}")

        # --- KET THUC FINE-TUNE CUA 1 STEP ---
        # Nap lai trong so tot nhat truoc khi sang lan Pruning tiep theo
        LOG_INFO(f"Reloading best weights from epoch with RMS {best_rms_in_step:.4f} for next pruning step.")
        model.load_state_dict(torch.load(temp_best_ckpt))
        
        # ==========================================
        # LUU CHECKPOINT CHINH THUC - BAO GOM CA MODEL OBJECT
        # ==========================================
        final_step_path = os.path.join(opts.model_dir, f"{opts.name}_step{step}_final.pth")
        torch.save({
            'model': model,                      # <-- LUU CA MODEL OBJECT DA PRUNE
            'net_state_dict': model.state_dict(),
            'net_opts': opts.net_opts,
            'data_opts': opts.data_opts,         # <-- Them data_opts de tien loi
            'rms': best_rms_in_step,
            'pruning_step': step + 1,
            'pruning_ratio': args.pruning_ratio,
            'original_params': original_params,
            'current_params': current_params,
        }, final_step_path)
        LOG_INFO(f"Saved checkpoint with FULL MODEL to: {final_step_path}")

    # ==========================================
    # LUU CHECKPOINT CUOI CUNG
    # ==========================================
    final_params = count_parameters(model)
    final_reduction = (1 - final_params / original_params) * 100
    
    final_model_path = os.path.join(opts.model_dir, f"{opts.name}_final.pth")
    torch.save({
        'model': model,                          # <-- LUU CA MODEL OBJECT DA PRUNE
        'net_state_dict': model.state_dict(),
        'net_opts': opts.net_opts,
        'data_opts': opts.data_opts,
        'rms': best_rms_in_step,
        'total_pruning_steps': args.pruning_steps,
        'pruning_ratio': args.pruning_ratio,
        'original_params': original_params,
        'final_params': final_params,
        'reduction_percent': final_reduction,
    }, final_model_path)
    
    LOG_INFO(f"\nIterative Pruning Complete!")
    LOG_INFO(f"Original params: {original_params:,}")
    LOG_INFO(f"Final params: {final_params:,}")
    LOG_INFO(f"Reduction: {final_reduction:.1f}%")
    LOG_INFO(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
