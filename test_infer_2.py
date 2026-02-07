import os
import numpy as np
import cv2
from ocamcamera import OcamCamera

# ================== CONFIG ==================
# Đường dẫn file ảnh đầu vào (Panorama Depth)
input_image_path = r"F:\algo\mvs_v119\video_inference_results_depth_only\depth_00008.png"

# Đường dẫn file Mask (Mask cho ảnh Fisheye đầu ra)
mask_image_path = r"F:\algo\mvs_v119\mask.png"

# Đường dẫn file ảnh đầu ra
output_image_path = r"F:\algo\mvs_v119\result_fisheye_masked.png"

# File OCamCalib
ocam_file = r"F:\algo\mvs_v119\ocam1_hyp.txt"

# Size fisheye mong muốn (Output)
fisheye_W = 800 
fisheye_H = 768

# Size panorama (dùng để tính toán map)
# Tốt nhất là để khớp với size ảnh input
# Nếu set None, code sẽ tự lấy theo ảnh input
pano_W = None 
pano_H = None
# ===========================================

def create_inverse_map(ocam_model, f_w, f_h, p_w, p_h):
    """
    Tạo bản đồ ánh xạ từ pixel Fisheye -> pixel Panorama
    """
    print("Precomputing inverse map (Fisheye -> Pano)...")
    
    # Tạo lưới tọa độ cho ảnh Fisheye đích
    xx, yy = np.meshgrid(np.arange(f_w), np.arange(f_h))
    points2D = np.stack([xx.flatten(), yy.flatten()], axis=0)

    # 1. Back-projection: Từ pixel 2D fisheye -> vector 3D world (Ray)
    rays = ocam_model.cam2world(points2D)
    X, Y, Z = rays

    # 2. Convert Cartesian -> Spherical
    norm = np.sqrt(X*X + Y*Y + Z*Z)
    
    # Theta (Latitude): [-pi/2, pi/2]
    theta = np.arcsin(Y / norm) 
    
    # Phi (Longitude): [-pi, pi]
    phi = np.arctan2(X, Z)

    # 3. Spherical -> Pixel Panorama (U, V)
    u = (phi + np.pi) / (2 * np.pi) * p_w
    v = (theta + np.pi / 2) / np.pi * p_h

    # Reshape về kích thước ảnh
    mapx = u.reshape(f_h, f_w).astype(np.float32)
    mapy = v.reshape(f_h, f_w).astype(np.float32)
    
    return mapx, mapy

def apply_mask(image, mask_path, target_size):
    """
    Hàm xử lý và áp dụng mask
    Logic: Đen (0) = Lấy, Trắng (255) = Bỏ
    """
    w, h = target_size
    
    if not os.path.exists(mask_path):
        print(f"Warning: Không tìm thấy mask tại {mask_path}. Bỏ qua bước mask.")
        return image

    print(f"Applying mask from: {mask_path}")
    # Đọc mask dạng Grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print("Error: Đọc file mask thất bại.")
        return image

    # Resize mask về đúng kích thước ảnh Output (Fisheye)
    # Dùng INTER_NEAREST để giữ nguyên giá trị 0 và 255 rạch ròi, tránh bị viền mờ
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Xử lý logic theo yêu cầu:
    # Yêu cầu: "Lấy depth trong mask (màu đen), ngoài mask (màu trắng) thì bỏ"
    # OpenCV mask hoạt động: 255 (Trắng) là giữ, 0 (Đen) là bỏ.
    # => Cần ĐẢO NGƯỢC mask (Invert)
    
    # Tạo mask nhị phân: Biến pixel < 128 (Đen) thành 255 (Trắng - để giữ lại)
    # Biến pixel > 128 (Trắng) thành 0 (Đen - để loại bỏ)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)

    # Áp dụng mask
    # Pixel nào binary_mask=255 sẽ giữ nguyên giá trị ảnh gốc
    # Pixel nào binary_mask=0 sẽ thành màu đen (0)
    masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
    
    return masked_image

def main():
    # 1. Kiểm tra file input
    if not os.path.exists(input_image_path):
        print(f"Error: Không tìm thấy ảnh tại {input_image_path}")
        return

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # 2. Load OCam
    print(f"Loading OCam model form {ocam_file}...")
    ocam = OcamCamera(ocam_file, fov=360)

    # 3. Đọc ảnh Panorama Depth
    print(f"Reading panorama: {input_image_path}")
    pano = cv2.imread(input_image_path) # Depth thường là ảnh 1 kênh hoặc 3 kênh giống nhau
    if pano is None:
        print("Failed to read image.")
        return

    real_h, real_w = pano.shape[:2]
    
    # Tự động lấy size nếu config là None
    p_w = pano_W if pano_W is not None else real_w
    p_h = pano_H if pano_H is not None else real_h

    # 4. Tính toán Map
    # Lưu ý: Map được tính dựa trên kích thước thật của ảnh Pano để đảm bảo đúng tỷ lệ
    mapx, mapy = create_inverse_map(ocam, fisheye_W, fisheye_H, p_w, p_h)

    # 5. Remap (Pano -> Fisheye)
    print("Remapping to fisheye view...")
    # Nếu ảnh input khác size config pano_W/H thì resize map hoặc ảnh (ở đây map tính theo size thật nên ok)
    fisheye = cv2.remap(
        pano,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0) 
    )

    # 6. ÁP DỤNG MASK
    fisheye = apply_mask(fisheye, mask_image_path, (fisheye_W, fisheye_H))

    # 7. Lưu kết quả
    cv2.imwrite(output_image_path, fisheye)
    print(f"Saved masked fisheye image to: {output_image_path}")

if __name__ == "__main__":
    main()