import argparse
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Count parameters from a pruned checkpoint")
    parser.add_argument(
        "--ckpt",
        default="checkpoints/romnistereo32_v21_bs8_prune_final.pth",
        help="Path to train_prune_v2.py checkpoint",
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


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        sys.exit(f"Checkpoint not found: {args.ckpt}")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    print(f"Checkpoint: {args.ckpt}")

    if isinstance(checkpoint, dict):
        if "original_params" in checkpoint:
            print(f"original_params: {checkpoint['original_params']}")
        if "final_params" in checkpoint:
            print(f"final_params: {checkpoint['final_params']}")
        if "model" in checkpoint:
            model = checkpoint["model"]
            total, trainable = count_params(model)
            print(f"model_total_params: {total}")
            print(f"model_trainable_params: {trainable}")
        elif "net_state_dict" in checkpoint:
            print("model object not found; only net_state_dict present")
    else:
        print("Checkpoint is not a dict; cannot parse")


if __name__ == "__main__":
    main()
