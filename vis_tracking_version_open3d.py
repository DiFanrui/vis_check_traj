import numpy as np
import open3d as o3d

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    输入：
    q - 四元数，格式为 [qx, qy, qz, qw]
    
    输出：
    R - 旋转矩阵 3x3
    """
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def read_poses(file_path, sep=',', matrix_shape=None, trajectory_indices=(slice(None), slice(None), 3), orientation_indices=None):
    """
    通用函数读取位姿数据并提取轨迹点和相机朝向。
    
    参数:
        file_path (str): 文件路径。
        sep (str): 数据分隔符，默认为 ','。
        matrix_shape (tuple): 数据的形状，例如 (4, 4) 表示 4x4 矩阵。
        trajectory_indices (tuple): 提取轨迹点的索引，例如 (slice(None), slice(None), 3) 表示提取 4x4 矩阵的最后一列。
        orientation_indices (tuple): 提取相机朝向的索引，例如 (slice(None), slice(0, 3), slice(0, 3)) 表示提取 3x3 旋转矩阵。
    
    返回:
        trajectory (np.ndarray): 提取的轨迹点。
        orientations (np.ndarray): 提取的相机朝向（旋转矩阵或四元数）。
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过以 '#' 开头的注释行
            if line.startswith('#'):
                continue
            if line.strip():
                pose = np.fromstring(line, sep=sep).reshape(matrix_shape)
                poses.append(pose)
    
    poses = np.array(poses)  # 转换为数组
    trajectory = poses[trajectory_indices]  # 提取轨迹点
    orientations = None
    if orientation_indices is not None:
        orientations = poses[orientation_indices]  # 提取相机朝向
    return trajectory, orientations

def plot_trajectory_open3d(file_path, file_type="matrix"):
    """
    使用 Open3D 绘制轨迹和相机朝向。
    
    参数:
        file_path (str): 文件路径。
        file_type (str): 文件类型，"matrix" 或 "quaternion"。
    """
    # 根据路径类型设置解析参数
    if file_type == "matrix":
        sep = ','
        matrix_shape = (4, 4)
        trajectory_indices = (slice(None), slice(0, 3), 3)  # 提取 4x4 矩阵的最后一列 (x, y, z)
        orientation_indices = (slice(None), slice(0, 3), slice(0, 3))  # 提取左上角 3x3 旋转矩阵
    elif file_type == "quaternion":
        sep = ' '
        matrix_shape = (8,)  # 每行是一个 8 元素的数组
        trajectory_indices = (slice(None), slice(1, 4))  # 提取第 1 到第 3 列 (x, y, z)
        orientation_indices = (slice(None), slice(4, 8))  # 提取第 4 到第 7 列 (qx, qy, qz, qw)
    else:
        raise ValueError("Unsupported file_type. Use 'matrix' or 'quaternion'.")

    # 读取轨迹数据和相机朝向
    trajectory, orientations = read_poses(
        file_path, sep=sep, matrix_shape=matrix_shape,
        trajectory_indices=trajectory_indices, orientation_indices=orientation_indices
    )

    # 创建 Open3D 点云对象
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(trajectory)
    points.paint_uniform_color([1, 0, 0])  # 红色表示轨迹点

    # 创建相机朝向的坐标系
    camera_frames = []
    for i in range(len(trajectory)):
        position = trajectory[i]
        if file_type == "matrix":
            # 从旋转矩阵中提取朝向
            rotation_matrix = orientations[i]
        elif file_type == "quaternion":
            # 将四元数转换为旋转矩阵
            q = orientations[i]
            rotation_matrix = quaternion_to_rotation_matrix(q)
        else:
            continue

        # 创建相机坐标系
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        camera_frame.rotate(rotation_matrix, center=(0, 0, 0))
        camera_frame.translate(position)
        camera_frames.append(camera_frame)

    # 可视化轨迹和相机朝向
    o3d.visualization.draw_geometries([points] + camera_frames)

if __name__ == "__main__":
    # 文件路径
    file_path = "lung_dataset/groundtruth.txt"
    # 文件类型："quaternion" 或 "matrix"
    file_type = "quaternion"  # 修改为 "matrix" 或 "quaternion" 根据需要选择

    # 绘制轨迹
    plot_trajectory_open3d(file_path, file_type)