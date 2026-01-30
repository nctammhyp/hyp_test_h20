import cv2

# Đường dẫn ảnh
path1 = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable\omnithings\mask1.png"
path2 = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable\omnithings\mask2.png"
path3 = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable\omnithings\mask3.png"

# Đọc ảnh grayscale
mask1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
mask3 = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)

# Resize (width=400, height=384)
mask1 = cv2.resize(mask1, (400, 384), interpolation=cv2.INTER_NEAREST)
mask2 = cv2.resize(mask2, (400, 384), interpolation=cv2.INTER_NEAREST)
mask3 = cv2.resize(mask3, (400, 384), interpolation=cv2.INTER_NEAREST)

import cv2
import os

save_dir = r"F:\Full-Dataset\hyp_data\hyp_data_01\hyp_data_01_trainable\omnithings_resized"
os.makedirs(save_dir, exist_ok=True)

cv2.imwrite(os.path.join(save_dir, "mask1.png"), mask1)
cv2.imwrite(os.path.join(save_dir, "mask2.png"), mask2)
cv2.imwrite(os.path.join(save_dir, "mask3.png"), mask3)
