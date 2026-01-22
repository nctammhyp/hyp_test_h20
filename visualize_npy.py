import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--pred", type=str, default=r"F:\algo\mvs_v119\results\hyp_01\invdepth_00001_romnistereo32_v6_bs32_e37.npy")
parser.add_argument("--gt", type=str, default=r"F:\algo\mvs_v119\omnidata\hyp_01\gt\00001.npy")
parser.add_argument("--cmap", type=str, default="plasma")
args = parser.parse_args()


pred = np.load(args.pred)
gt = np.load(args.gt)

# ===== RESIZE GT TO PRED =====
H_p, W_p = pred.shape[:2]
H_g, W_g = gt.shape[:2]

if (H_g, W_g) != (H_p, W_p):
    gt = cv2.resize(gt, (W_p, H_p), interpolation=cv2.INTER_NEAREST)

print("Pred size:", pred.shape)
print("GT size (resized):", gt.shape)

# ===== CLEAN =====
pred_clean = pred.copy()
pred_clean[~np.isfinite(pred_clean)] = np.nan

gt_clean = gt.copy()
gt_clean[~np.isfinite(gt_clean)] = np.nan

# ===== COMMON SCALE =====
vmin = np.nanmin(gt_clean)
vmax = np.nanmax(gt_clean)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(pred_clean, cmap=args.cmap, vmin=vmin, vmax=vmax)
axes[0].set_title("Pred (raw)")
axes[0].axis("off")

axes[1].imshow(gt_clean, cmap=args.cmap, vmin=vmin, vmax=vmax)
axes[1].set_title("GT (resized)")
axes[1].axis("off")

# ===== CLICK HANDLER =====
def onclick(event):
    if event.inaxes != axes[0]:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    if y < 0 or y >= pred.shape[0] or x < 0 or x >= pred.shape[1]:
        return

    print(
        f"[Click] (x={x}, y={y}) | "
        f"Pred={pred[y, x]} | GT={gt[y, x]}"
    )

fig.canvas.mpl_connect("button_press_event", onclick)

plt.tight_layout()
plt.show()
