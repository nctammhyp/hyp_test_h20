import cv2
import os

# Thư mục chứa ảnh
img_dir = r"F:\algo\mvs_v119\combined_inference_results"
output_video = "output.mp4"
fps = 15  # số frame / giây

# Lấy danh sách file png và sort theo tên
images = sorted([
    img for img in os.listdir(img_dir)
    if img.lower().endswith(".png")
])

if not images:
    raise ValueError("Không tìm thấy file .png nào")

# Đọc frame đầu để lấy kích thước
first_frame = cv2.imread(os.path.join(img_dir, images[0]))
height, width, _ = first_frame.shape

# Khởi tạo video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Ghi từng frame
for img_name in images:
    img_path = os.path.join(img_dir, img_name)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print("Done! Video saved as", output_video)
