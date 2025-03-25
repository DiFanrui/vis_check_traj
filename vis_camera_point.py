import pickle
import numpy as np
import open3d as o3d
import json

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R

def load_poses_from_quaternion(txt_file):
    """从 txt 文件加载相机位姿，返回帧数和对应的转换矩阵（4x4）"""
    poses_dict = {}  # 用于存储帧数和对应的转换矩阵（字典形式）

    with open(txt_file, 'r') as f:
        for line in f:
            # 跳过以 '#' 开头的注释行
            if line.startswith('#'):
                continue
            
            # 读取每一行数据，去除两端空白符并拆分为多个部分
            data = line.strip().split()
            
            # 如果数据不完整，跳过
            if len(data) != 8:
                continue
            
            # 解析数据
            frame_id = int(float(data[0]))  # 时间戳作为帧索引
            tx, ty, tz = float(data[1]), float(data[2]), float(data[3])
            qx, qy, qz, qw = float(data[4]), float(data[5]), float(data[6]), float(data[7])
            
            # 将四元数转换为旋转矩阵
            R = quaternion_to_rotation_matrix([qx, qy, qz, qw])
            
            # 将平移向量和旋转矩阵组合成 4x4 转换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = np.array([tx, ty, tz])
            
            # 存储帧索引和对应的转换矩阵
            poses_dict[frame_id] = transform_matrix  # 以帧ID为键，转换矩阵为值

    return poses_dict  # 返回字典格式

def create_camera_frame(origin, corners, center, color):
    """创建 Open3D 线框相机表示"""
    points = o3d.utility.Vector3dVector(np.vstack((origin, corners, center)))

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 光心到矩形四个角
        [1, 2], [2, 3], [3, 4], [4, 1],  # 连接矩形四条边
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set

def load_points(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_camera_poses_and_points_to_ply(poses_dict, points, size=1.0, length=0.5, width=0.4, depth=0.3, output_file="camera_poses_with_points.ply", max_points=600):
    """
    将相机位姿和前六百个点保存为 ply 文件。
    :param poses_dict: 相机位姿字典，键为帧ID，值为4x4变换矩阵
    :param points: 点云数据
    :param colors: 点云颜色数据
    :param size: 相机框的缩放比例
    :param length: 矩形长边
    :param width: 矩形短边
    :param depth: 相机光心到矩形平面的距离
    :param output_file: 输出的 ply 文件路径
    """
    all_points = []
    all_lines = []
    all_colors = []

    point_offset = 0  # 用于调整点索引

    # 添加点云数据到 all_points, all_colors
    point_ids = list(points.keys())[:max_points]  # 只取前600个点
    for point_id in point_ids:
        point_list = points[point_id]
        point_xyz = point_list[0]  # 三维坐标
        point_color = point_list[1]  # 颜色信息

        all_points.append(point_xyz)
        all_colors.append(point_color)

    for frame_id, T in poses_dict.items():
        R_mat = T[:3, :3]  # 旋转矩阵
        t_vec = T[:3, 3]   # 平移向量

        # 计算相机光心和四个角点
        origin = np.array([0, 0, depth]) * size
        corners = np.array([
            [-length/2, -width/2, 0],
            [length/2, -width/2, 0],
            [length/2, width/2, 0],
            [-length/2, width/2, 0]
        ]) * size
        center = np.array([0, 0, 0])  # 矩形中心点

        # 变换到世界坐标系
        origin = R_mat @ origin + t_vec
        corners = (R_mat @ corners.T).T + t_vec
        center = R_mat @ center + t_vec

        # 创建相机框
        camera_frame = create_camera_frame(origin, corners, center, [1, 0, 0])  # 红色框
        
        # 获取线框的点、线和颜色信息
        all_points.extend(np.asarray(camera_frame.points))
        all_lines.extend(np.asarray(camera_frame.lines) + point_offset)  # 调整点索引
        all_colors.extend(np.asarray(camera_frame.colors))
        
        # 更新点偏移量
        point_offset += len(camera_frame.points)

    # 创建一个包含所有相机框的 LineSet
    combined_line_set = o3d.geometry.LineSet()
    combined_line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
    combined_line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines))
    combined_line_set.colors = o3d.utility.Vector3dVector(np.array(all_colors))

    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(all_points[:max_points]))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(all_colors[:max_points]))

    # 合并点云和相机框
    combined_geometry = combined_line_set + point_cloud
    
    # 保存为 ply 文件
    o3d.io.write_line_set(output_file, combined_line_set)
    print(f"相机位姿和点云已保存到 {output_file}")

# 你的 pkl 文件路径
camera_file = "/home/data1/difanrui/Project/vis_check_traj/dataset/groundtruth.txt"
point_file = "/home/data1/difanrui/Project/vis_check_traj/points.json"  # 点云数据文件路径

# 加载相机位姿
poses_dict = load_poses_from_quaternion(camera_file)

# 加载点云数据
points_data = load_points(point_file)

# 保存相机位姿和前六百个点到 ply 文件
save_camera_poses_and_points_to_ply(poses_dict, points_data, size=1.0, length=0.5, width=0.4, depth=0.3, output_file="ply_file/combined_camera_poses_with_points.ply", max_points=600)
