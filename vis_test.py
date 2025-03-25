import pickle
import json
import numpy as np
import open3d as o3d

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
def visualize_trajectory_and_points(pkl_file, json_file, max_points=2, max_frames=20):
    # 加载数据
    frame_indices, transform_matrices = load_poses(pkl_file)
    points_data = load_points(json_file)
    
    # 相机轨迹（蓝色）
    traj_points = []

    # 三维点（红色）
    point_cloud = o3d.geometry.PointCloud()
    point_colors = []
    i = 0
    # 遍历所有帧，提取相机轨迹
    for frame in frame_indices:  # 遍历所有帧
        if i >= max_frames:
            break
        T_world_cam = transform_matrices[frame]
        
        # 排除单位矩阵（非关键帧）
        if np.allclose(T_world_cam, np.eye(4)):  
            continue  # 直接跳过
        
        # 提取相机中心位置
        traj_points.append(T_world_cam[:3, 3])
        i += 1

    # 遍历前max_points个点，提取三维点位置和颜色信息
    point_ids = list(points_data.keys())[:max_points]  # 只取前600个点
    for point_id in point_ids:
        point_list = points_data[point_id]
        # 提取点的三维坐标和颜色信息
        point_xyz = point_list[0]  # 三维坐标
        point_color = point_list[1]  # 颜色信息

        point_cloud.points.append(point_xyz)
        point_colors.append(point_color)

    # 设置点云颜色
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(point_colors))

    # 创建相机轨迹线
    traj_line_set = o3d.geometry.LineSet()
    traj_line_set.points = o3d.utility.Vector3dVector(np.array(traj_points))
    traj_lines = [[i, i + 1] for i in range(len(traj_points) - 1)]
    traj_line_set.lines = o3d.utility.Vector2iVector(traj_lines)
    traj_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(traj_lines))  # 蓝色

    # 可视化
    o3d.visualization.draw_geometries([point_cloud, traj_line_set])

# === 运行可视化 ===
pkl_file = "poses.pickle"
json_file = "points.json"
visualize_trajectory_and_points(pkl_file, json_file, max_points=2, max_frames=20)