import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

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
# 读取彩色图像（确保路径正确）
# ------------------------------
image_file = "check_distorted/0000.png"   # 请确保路径正确

img = Image.open(image_file)  # 不进行灰度转换
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
# 计算两张图像的差异
# ------------------------------
difference = np.abs(img_np.astype(np.int32) - undistorted.astype(np.int32))  # 计算绝对差值
difference = np.clip(difference, 0, 255).astype(np.uint8)  # 将差值限制在 [0, 255] 范围内

output_dir = "check_distorted/"  # 保存路径

# 保存原始图像
plt.imsave(os.path.join(output_dir, "distorted_image.png"), img_np)

# 保存去畸变图像
plt.imsave(os.path.join(output_dir, "undistorted_image.png"), undistorted)

# 保存差异图像
plt.imsave(os.path.join(output_dir, "difference_image.png"), difference)

# ------------------------------
# 显示原始图像、去畸变图像和差异图像
# ------------------------------

plt.figure(figsize=(18, 6))
# 显示原始图像
plt.subplot(1, 3, 1)
plt.title("Distorted Image")
plt.imshow(img_np)
plt.axis("off")

# 显示去畸变图像
plt.subplot(1, 3, 2)
plt.title("Undistorted Image")
plt.imshow(undistorted)
plt.axis("off")

# 显示差异图像
plt.subplot(1, 3, 3)
plt.title("Difference Image")
plt.imshow(difference)
plt.axis("off")

plt.show()

