import pickle
import json
import numpy as np

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

# === 3. 保存为 PLY 文件 ===
def save_to_ply(ply_file, points, colors, camera_trajectory=None):
    """
    将点云和相机轨迹保存为 PLY 文件。
    :param ply_file: 输出的 PLY 文件路径
    :param points: 点云坐标 (N, 3) 的 NumPy 数组
    :param colors: 点云颜色 (N, 3) 的 NumPy 数组，颜色值范围为 [0, 255]
    :param camera_trajectory: 相机轨迹坐标 (M, 3) 的 NumPy 数组
    """
    # 确保点云和颜色的形状匹配
    assert points.shape[0] == colors.shape[0], "点云和颜色的数量不匹配"

    # 写入 PLY 文件
    with open(ply_file, 'w') as f:
        # 写入 PLY 文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(points.shape[0] + camera_trajectory.shape[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write("element edge {}\n".format(camera_trajectory.shape[0] - 1))
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write("end_header\n")

        # 写入点云数据
        for i in range(points.shape[0]):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                    f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")

        # 写入相机轨迹数据
        for i in range(camera_trajectory.shape[0]):
            # 相机轨迹用蓝色表示
            f.write(f"{camera_trajectory[i, 0]:.6f} {camera_trajectory[i, 1]:.6f} {camera_trajectory[i, 2]:.6f} "
                    f"0 0 255\n")

        # 写入相机轨迹的边信息
        for i in range(camera_trajectory.shape[0] - 1):
            f.write(f"{points.shape[0] + i} {points.shape[0] + i + 1} 0 0 255\n")

# === 4. 可视化相机轨迹和三维点 ===
def visualize_trajectory_and_points(pkl_file, json_file, ply_file):
    # 加载数据
    frame_indices, transform_matrices = load_poses(pkl_file)
    points_data = load_points(json_file)

    # 相机轨迹
    camera_trajectory = []

    # 三维点和颜色
    points = []
    colors = []

    # 遍历所有帧，提取相机轨迹
    for frame in range(len(frame_indices)):
        T_world_cam = transform_matrices[frame]
        
        # 排除单位矩阵（非关键帧）
        if np.allclose(T_world_cam, np.eye(4)):  
            continue  # 直接跳过
        
        # 提取相机中心位置
        camera_trajectory.append(T_world_cam[:3, 3])

    # 遍历所有点，提取三维点位置和颜色信息，只处理前600个点
    point_counter = 0
    for point_id, point_list in points_data.items():
        if point_counter >= 600:  # 只处理前600个点
            break
        # 提取点的三维坐标和颜色信息
        point_xyz = point_list[0]  # 三维坐标
        point_color = point_list[1]  # 颜色信息

        points.append(point_xyz)
        colors.append(point_color)

        point_counter += 1

    # 转换为 NumPy 数组
    points = np.array(points)
    colors = np.array(colors) * 255  # 将颜色范围从 [0, 1] 转换为 [0, 255]
    camera_trajectory = np.array(camera_trajectory)

    # 保存为 PLY 文件
    save_to_ply(ply_file, points, colors, camera_trajectory)

# === 运行 ===
pkl_file = "/home/data1/difanrui/Project/OneSLAM/experiments/exp_03182025_10:05:21/poses.pickle"
json_file = "/home/data1/difanrui/Project/OneSLAM/experiments/exp_03182025_10:05:21/points.json"
ply_file = "/home/data1/difanrui/Project/OneSLAM/experiments/exp_03182025_10:05:21/output_first2frame.ply"
visualize_trajectory_and_points(pkl_file, json_file, ply_file)