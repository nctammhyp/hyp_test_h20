import cv2
import numpy as np

# ========= PATH (Giữ nguyên) =========
cam1_path = r"F:\algo\mvs_v119\omnidata\hyp_sync_1\cam1\00002.png"
cam2_path = r"F:\algo\mvs_v119\omnidata\hyp_sync_1\cam2\00002.png"
cam3_path = r"F:\algo\mvs_v119\omnidata\hyp_sync_1\cam3\00002.png"
gt_path   = r"F:\algo\mvs_v119\omnidata\hyp_sync_1\gt\00002.tiff"
pred_path = r"F:\algo\mvs_v119\results\hyp_sync_1\invdepth_00001_romnistereo32_v6_bs32_e1.png"

max_depth_view = 30.0
# Resize nhỏ lại để 5 ảnh đứng cạnh nhau không bị quá dài
DISPLAY_W = 400 
DISPLAY_H = 250

# ========= LOAD =========
cam1 = cv2.imread(cam1_path)
cam2 = cv2.imread(cam2_path)
cam3 = cv2.imread(cam3_path)
gt   = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)

if any(x is None for x in [cam1, cam2, cam3, gt, pred]):
    print("❌ Lỗi load ảnh")
    exit()

# Chuyển kiểu dữ liệu
gt = gt.astype(np.float32)
pred = pred.astype(np.float32)


# print("GT dtype:", gt.dtype, "min:", gt.min(), "max:", gt.max())
# print("Pred dtype:", pred.dtype, "min:", pred.min(), "max:", pred.max())


gt[np.isnan(gt)] = 0
pred[np.isnan(pred)] = 0

# !!! QUAN TRỌNG: Resize dữ liệu gốc để khớp với tọa độ hiển thị !!!
gt = cv2.resize(gt, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
pred = cv2.resize(pred, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)

# ========= COLORIZE =========
def colorize(depth):
    clipped = np.clip(depth, 0, max_depth_view)
    norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

gt_color = colorize(gt)
pred_color = colorize(pred)

# Resize các ảnh camera
cam1 = cv2.resize(cam1, (DISPLAY_W, DISPLAY_H))
cam2 = cv2.resize(cam2, (DISPLAY_W, DISPLAY_H))
cam3 = cv2.resize(cam3, (DISPLAY_W, DISPLAY_H))
gt_color = cv2.resize(gt_color, (DISPLAY_W*3, DISPLAY_H))
pred_color = cv2.resize(pred_color, (DISPLAY_W*3, DISPLAY_H))

# ========= STACK =========
# 5 ảnh x 400px = 2000px (Vừa khít hoặc hơi to hơn màn hình Full HD một chút)
vis = np.vstack([np.hstack([cam1, cam2, cam3]), gt_color, pred_color])

# ========= CLICK EVENT =========
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        temp = vis.copy()

        # Xác định row
        if DISPLAY_H <= y < 2 * DISPLAY_H:
            region = "GT"
            src_y = y - DISPLAY_H
        elif 2 * DISPLAY_H <= y < 3 * DISPLAY_H:
            region = "Pred"
            src_y = y - 2 * DISPLAY_H
        else:
            return  # Không click vào cam thì bỏ qua

        # Vì GT và Pred bị resize ngang x3
        src_x = int(x / 3)

        if src_x < 0 or src_x >= DISPLAY_W or src_y < 0 or src_y >= DISPLAY_H:
            return

        gt_val = gt[src_y, src_x]
        pred_val = pred[src_y, src_x]
        diff = abs(gt_val - pred_val)

        print(f"[{region}] Pos(x={src_x}, y={src_y}) | GT={gt_val:.4f} | Pred={pred_val:.4f} | Diff={diff:.4f}")

        cv2.circle(temp, (x, y), 5, (0, 255, 0), -1)
        text = f"G:{gt_val:.2f} P:{pred_val:.2f}"
        cv2.putText(temp, text, (x - 60, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("Viewer", temp)


# ========= SHOW =========
print(f"Total size: {vis.shape[1]}x{vis.shape[0]}")
print("Click vào vùng GT hoặc Pred để xem giá trị")

cv2.namedWindow("Viewer", cv2.WINDOW_AUTOSIZE) # Dùng AUTOSIZE để nó không tự ép tỉ lệ
cv2.setMouseCallback("Viewer", click_event)
cv2.imshow("Viewer", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()