import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# ------------------------------
# 畸变参数和内参 (与 C++ 代码一致)
# ------------------------------
# 畸变参数：k1, k2, p1, p2
k1 = 0.0237323472870419
k2 = -0.320432607152857
p1 = 0.00019359
p2 = 0

# 相机内参
fx = 447.679079249249
fy = 448.768431364843
cx = 252.560061902772
cy = 233.658637132913

# ------------------------------
# 输入和输出目录
# ------------------------------
input_dir = "lung_dataset/images"  # 输入图像目录
output_dir = "lung_dataset/undistorted_images"  # 输出图像目录
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

# ------------------------------
# 批处理图像
# ------------------------------
for image_file in os.listdir(input_dir):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue  # 跳过非图像文件

    input_path = os.path.join(input_dir, image_file)
    output_path = os.path.join(output_dir, image_file)

    # 读取彩色图像
    img = Image.open(input_path)
    img_np = np.array(img)
    rows, cols, _ = img_np.shape  # 获取彩色图像的维度

    # ------------------------------
    # 构建像素网格 (u, v)
    # ------------------------------
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))

    # ------------------------------
    # 计算归一化坐标 (x, y)
    # ------------------------------
    x = (u - cx) / fx
    y = (v - cy) / fy

    # ------------------------------
    # 计算径向距离 r 并根据公式计算畸变后的坐标
    # ------------------------------
    r = np.sqrt(x**2 + y**2)

    # 根据公式计算去畸变前的归一化坐标（x_distorted, y_distorted）
    x_distorted = x * (1 + k1 * r**2 + k2 * r**4) + 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
    y_distorted = y * (1 + k1 * r**2 + k2 * r**4) + p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y

    # 将去畸变后的归一化坐标映射回像素坐标 (u_distorted, v_distorted)
    u_distorted = fx * x_distorted + cx
    v_distorted = fy * y_distorted + cy

    # ------------------------------
    # 最近邻插值：将每个像素位置对应到畸变图像上的位置
    # ------------------------------
    u_distorted_nn = np.round(u_distorted).astype(np.int32)
    v_distorted_nn = np.round(v_distorted).astype(np.int32)

    # 创建一个与原图尺寸相同的空白图像，用于存放去畸变后的结果
    undistorted = np.zeros_like(img_np)

    # 只对落在图像范围内的像素赋值
    mask = (u_distorted_nn >= 0) & (u_distorted_nn < cols) & (v_distorted_nn >= 0) & (v_distorted_nn < rows)
    undistorted[mask] = img_np[v_distorted_nn[mask], u_distorted_nn[mask]]

    # ------------------------------
    # 保存去畸变后的图像
    # ------------------------------
    undistorted_img = Image.fromarray(undistorted)
    undistorted_img.save(output_path)

    print(f"已处理并保存图像：{output_path}")

print("批处理完成！所有去畸变图像已保存到 lung_dataset/undistorted_images 目录中。")