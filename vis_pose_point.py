import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

# === 1. 解析 pose.pkl ===
def load_poses(pkl_file):
    with open(pkl_file, 'rb') as f:
        pose_data = pickle.load(f)  # 反序列化
    frame_indices = list(pose_data.keys())  # 获取帧ID
    transform_matrices = {frame: pose_data[frame][0] for frame in frame_indices}  # 只取 4x4 矩阵
    return frame_indices, transform_matrices

# === 2. 解析 points.json ===
def load_points(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# === 3. 可视化相机轨迹和三维点 ===
def visualize_trajectory_and_points(pkl_file, json_file, max_points=600):
    # 加载数据
    frame_indices, transform_matrices = load_poses(pkl_file)
    points_data = load_points(json_file)

    # 相机轨迹（蓝色）
    traj_x, traj_y, traj_z = [], [], []

    # 三维点（红色）
    point_x, point_y, point_z = [], [], []
    point_colors = []  # 用于存储点的颜色信息

    # 遍历所有帧，提取相机轨迹
    for frame in frame_indices:  # 遍历所有帧
        T_world_cam = transform_matrices[frame]
        
        # 排除单位矩阵（非关键帧）
        if np.allclose(T_world_cam, np.eye(4)):  
            continue  # 直接跳过
        
        # 提取相机中心位置
        traj_x.append(T_world_cam[0, 3])
        traj_y.append(T_world_cam[1, 3])
        traj_z.append(T_world_cam[2, 3])

    # 遍历前max_points个点，提取三维点位置和颜色信息
    point_ids = list(points_data.keys())[:max_points]  # 只取前600个点
    for point_id in point_ids:
        point_list = points_data[point_id]
        # 提取点的三维坐标和颜色信息
        point_xyz = point_list[0]  # 三维坐标
        point_color = point_list[1]  # 颜色信息

        point_x.append(point_xyz[0])
        point_y.append(point_xyz[1])
        point_z.append(point_xyz[2])
        point_colors.append(point_color)

    # 绘制
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制相机轨迹
    ax.plot(traj_x, traj_y, traj_z, label="Camera Trajectory", color='blue', marker='o')
    
    # 绘制三维点
    # 将颜色信息从 [0, 1] 范围转换到 [0, 255] 范围
    point_colors = np.array(point_colors) * 255
    ax.scatter(point_x, point_y, point_z, c=point_colors / 255.0, label="3D Points", marker='o')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-100, 100])
    ax.set_title("Camera Trajectory & 3D Points Visualization")
    ax.legend()
    plt.show()

# === 运行可视化 ===
pkl_file = "/home/data1/difanrui/Project/OneSLAM/experiments/exp_03182025_10:05:21/poses.pickle"
json_file = "/home/data1/difanrui/Project/OneSLAM/experiments/exp_03182025_10:05:21/points.json"
visualize_trajectory_and_points(pkl_file, json_file, max_points=400)
