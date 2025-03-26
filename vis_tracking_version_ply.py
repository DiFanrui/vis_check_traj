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

def save_trajectory_as_pointcloud(file_path, file_type="matrix", output_path="trajectory_with_orientations.ply"):
    """
    将轨迹和相机朝向保存为点云文件。
    
    参数:
        file_path (str): 文件路径。
        file_type (str): 文件类型，"matrix" 或 "quaternion"。
        output_path (str): 输出点云文件路径。
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
    
    # 创建箭头表示相机朝向
    arrow_meshes = []
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

        # 创建箭头
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01,  # 动态调整箭头的圆柱半径
            cone_radius=0.03,      # 动态调整箭头的锥体半径
            cylinder_height=0.2,  # 动态调整箭头的圆柱高度
            cone_height=0.1       # 动态调整箭头的锥体高度
        )
        arrow.paint_uniform_color([0, 0, 1])  # 蓝色表示箭头
        arrow.rotate(rotation_matrix, center=(0, 0, 0))
        arrow.translate(position)
        arrow_meshes.append(arrow)

    # 合并轨迹点和箭头
    combined_mesh = o3d.geometry.TriangleMesh()
    for arrow in arrow_meshes:
        combined_mesh += arrow

    # 标记初始点和终止点
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    start_sphere.paint_uniform_color([0, 1, 0])  # 绿色表示初始点
    start_sphere.translate(trajectory[0])  # 移动到初始点位置

    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    end_sphere.paint_uniform_color([1, 0, 0])  # 红色表示终止点
    end_sphere.translate(trajectory[-1])  # 移动到终止点位置
    
    # 创建逐帧连接的轨迹线
    lines = [[i, i + 1] for i in range(len(trajectory) - 1)]  # 每两个点之间创建一条线
    for line in lines:
        start_point = trajectory[line[0]]
        end_point = trajectory[line[1]]
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)

        # 跳过长度为 0 的线段
        if line_length == 0:
            continue

        line_direction = line_vector / line_length

        # 创建细长的圆柱体表示轨迹线
        line_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.01,  # 轨迹线的半径
            height=line_length  # 轨迹线的长度
        )
        line_cylinder.paint_uniform_color([0.5, 0, 0.5])  # 紫色表示轨迹线

        # 计算旋转矩阵并应用旋转
        z_axis = np.array([0, 0, 1])  # 圆柱体默认沿 Z 轴
        rotation_axis = np.cross(z_axis, line_direction)
        rotation_angle = np.arccos(np.dot(z_axis, line_direction))
        if np.linalg.norm(rotation_axis) > 1e-6:  # 避免零向量
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
            line_cylinder.rotate(rotation_matrix, center=(0, 0, 0))

        # 平移到正确位置
        line_cylinder.translate(start_point + line_direction * (line_length / 2))  # 将圆柱体移动到两点之间
        combined_mesh += line_cylinder


    arrow_output_path = output_path.replace(".ply", "_arrows.ply")
    combined_mesh += start_sphere
    combined_mesh += end_sphere
    o3d.io.write_triangle_mesh(arrow_output_path, combined_mesh)
    print(f"相机朝向箭头已保存为点云文件：{arrow_output_path}")

if __name__ == "__main__":
    # 文件路径
    file_path = "lung_dataset/groundtruth.txt"
    # 文件类型："quaternion" 或 "matrix"
    file_type = "quaternion"  # 修改为 "matrix" 或 "quaternion" 根据需要选择
    # 输出点云文件路径
    output_path = "ply_file/trajectory_with_orientations.ply"

    # 保存轨迹和相机朝向为点云文件
    save_trajectory_as_pointcloud(file_path, file_type, output_path)