import cv2
import os

# 打开视频文件
video_path = 'stable_seq_005_part_1_dif_1.mp4'
output_dir = 'stable_seq_005_part_1_dif_1'

# 检查输出目录是否存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
desired_frames = 1884

# 计算抽帧间隔
frame_interval = max(1, total_frames // desired_frames)

# 读取视频帧
frame_count = 0
saved_frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 每隔 frame_interval 帧保存一帧
    if frame_count % frame_interval == 0 and saved_frame_count < desired_frames:
        frame_filename = os.path.join(output_dir, f'{saved_frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

# 释放视频捕获对象
cap.release()
print("Video frames extracted successfully.")