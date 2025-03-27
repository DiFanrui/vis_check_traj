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
            frame_num_str = str(frame_num)  # 确保 frame_num 是字符串
            if frame_num_str in image_files:
                image_path = image_files[frame_num_str]
                image = cv2.imread(image_path)
                if image is not None:
                    # 标记点
                    x, y = int(coords[0]), int(coords[1])
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 红色圆点
                    # 添加图像名称
                    image_name = os.path.basename(image_path)
                    cv2.putText(image, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # 保存标记后的图像
                    output_path = os.path.join(output_dir, f'image{frame_num_str}_marked.jpg')
                    cv2.imwrite(output_path, image)
                    marked_images.append(image)
                    print(f"已标记并保存图像：{output_path}")
                else:
                    print(f"无法读取图像：{image_path}")
            else:
                print(f"图像文件未找到：frame_num={frame_num_str}")
    return marked_images

def concatenate_images(images, gap=10):
    if not images:
        raise ValueError("输入的图像列表为空，无法拼接！")
    
    # 确保所有图像的高度一致
    max_height = max(image.shape[0] for image in images)
    padded_images = []
    for image in images:
        if image.shape[0] < max_height:
            # 在图像底部填充黑色像素
            padding = np.zeros((max_height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
            padded_image = np.vstack((image, padding))
        else:
            padded_image = image
        padded_images.append(padded_image)
    
    # 计算拼接后图像的尺寸
    total_width = sum(image.shape[1] for image in padded_images) + gap * (len(padded_images) - 1)
    concatenated_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    
    # 拼接图像
    current_x = 0
    for image in padded_images:
        concatenated_image[:image.shape[0], current_x:current_x + image.shape[1]] = image
        current_x += image.shape[1] + gap
    
    return concatenated_image

# 定义点的坐标
points = {
    "3": [
        [0, [170.0, 287.0]],
        [40, [176.5126953125, 296.0967712402344]],
        [80, [203.76791381835938, 323.6740417480469]]
    ]
}

# 图像文件路径
image_files = {
    "0": "lung_dataset/undistorted_images/00000000.jpg",
    "40": "lung_dataset/undistorted_images/00000040.jpg",
    "80": "lung_dataset/undistorted_images/00000080.jpg"
}

# 输出文件目录
output_dir = "marked_image/lung_undistored_marked"

# 在图像上标记点
marked_images = mark_points_on_images(points, image_files, output_dir)

if not marked_images:
    print("没有标记任何图像，请检查输入的点和图像文件路径！")
else:
    print("已在图像上标记点并保存到输出文件。")

    # 拼接图像
    concatenated_image = concatenate_images(marked_images)

    # 保存拼接后的图像
    output_path = os.path.join(output_dir, "concatenated_image.jpg")
    cv2.imwrite(output_path, concatenated_image)
    print(f"拼接后的图像已保存到：{output_path}")