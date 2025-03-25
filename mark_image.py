import cv2
import os
import numpy as np

def mark_points_on_images(points, image_files, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    marked_images = []
    for frame_id, point_data in points.items():
        for point_id, (frame_num, coords) in enumerate(point_data):
            if str(frame_num) in image_files:
                image_path = image_files[str(frame_num)]
                image = cv2.imread(image_path)
                if image is not None:
                    x, y = int(coords[0]), int(coords[1])
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 红色圆点
                    # 添加图像名称
                    image_name = os.path.basename(image_path)
                    cv2.putText(image, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    output_path = os.path.join(output_dir, f'image{frame_num}_marked.jpg')
                    marked_images.append(image)
                    cv2.imwrite(output_path, image)
                else:
                    print(f"无法读取图像：{image_path}")
    return marked_images

def concatenate_images(images, gap=10):
    # 计算拼接后图像的尺寸
    total_height = max(image.shape[0] for image in images)
    total_width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    
    # 创建拼接后的图像
    concatenated_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # 拼接图像
    current_x = 0
    for image in images:
        concatenated_image[:image.shape[0], current_x:image.shape[1] + current_x] = image
        current_x += image.shape[1] + gap
    
    return concatenated_image

# 定义点的坐标
points = {
    "3": [
        [0, [170.0, 287.0]],
        [1, [176.5126953125, 296.0967712402344]],
        [2, [203.76791381835938, 323.6740417480469]]
    ]
}

# 图像文件路径
image_files = {
    "0": "dataset/images/00000000.jpg",
    "1": "dataset/images/00000040.jpg",
    "2": "dataset/images/00000080.jpg"
}

# 输出文件目录
output_dir = "marked"

# 在图像上标记点
marked_images = mark_points_on_images(points, image_files, output_dir)

print("已在图像上标记点并保存到输出文件。")

# 拼接图像
concatenated_image = concatenate_images(marked_images)

# 保存拼接后的图像
output_path = os.path.join(output_dir, "concatenated_image.jpg")
cv2.imwrite(output_path, concatenated_image)