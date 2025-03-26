import numpy as np
import matplotlib.pyplot as plt

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    输入：
    q - 四元数，格式为 [qx, qy, qz, qw]
    
    输出：
    R - 旋转矩阵 3x3
    """
    qx, qy, qz, qw = q
    
    # 计算旋转矩阵的元素
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R

def read_poses(file_path, sep=',', matrix_shape=None, trajectory_indices=(slice(None), slice(None), 3), orientation_indices=None):
    """
    通用函数读取位姿数据并提取轨迹点。
    
    参数:
        file_path (str): 文件路径。
        sep (str): 数据分隔符，默认为 ','。
        matrix_shape (tuple): 数据的形状，例如 (4, 4) 表示 4x4 矩阵。
        trajectory_indices (tuple): 提取轨迹点的索引，例如 (slice(None), slice(None), 3) 表示提取 4x4 矩阵的最后一列。
    
    返回:
        trajectory (np.ndarray): 提取的轨迹点。
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

def plot_trajectory(file_path, file_type=None):
    """
    绘制轨迹的函数，根据文件类型选择不同的解析方式。
    
    参数:
        file_path (str): 文件路径。
        path_type (str): 文件类型，"matrix" 或 "quaternion"。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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
        raise ValueError("Unsupported path_type. Use 'matrix' or 'quaternion'.")

    # 读取轨迹数据和相机朝向
    trajectory, orientations = read_poses(
        file_path, sep=sep, matrix_shape=matrix_shape,
        trajectory_indices=trajectory_indices, orientation_indices=orientation_indices
    )

    # 绘制轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
    # 标注初始点和结束点
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=100, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', s=100, label='End')

    # 打印初始点和结束点的位置
    print("Start Point:", trajectory[0, 0], trajectory[0, 1], trajectory[0, 2])
    print("End Point:", trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2])

    # 绘制相机朝向
    if orientations is not None:
        for i in range(len(trajectory)):
            position = trajectory[i]
            if file_type == "matrix":
                # 从旋转矩阵中提取朝向向量（例如 x 轴方向）
                orientation = orientations[i, :, 0]  # 提取第 1 列作为朝向
            elif file_type == "quaternion":
                # 将四元数转换为旋转矩阵
                q = orientations[i]
                orientation = quaternion_to_rotation_matrix(q)  # 自定义函数，将四元数转换为方向向量
            else:
                continue

            # 绘制朝向箭头
            ax.quiver(
                position[0], position[1], position[2],  # 起点
                orientation[0], orientation[1], orientation[2],  # 朝向向量
                length=0.5, color='blue', normalize=True
            )
    
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Visualization')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    # 文件路径
    file_path = "lung_dataset/groundtruth.txt"
    # 文件类型："quaternion" 或 "matrix"
    file_type = "quaternion"  # 修改为 "matrix" 或 "quaternion" 根据需要选择

    # 绘制轨迹
    plot_trajectory(file_path, file_type)