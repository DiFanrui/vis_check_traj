import numpy as np
import matplotlib.pyplot as plt

def read_poses(file_path, sep=',', matrix_shape=None, trajectory_indices=(slice(None), slice(None), 3)):
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
    return trajectory

def plot_trajectory(file_path, path_type="matrix"):
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
    elif file_type == "quaternion":
        sep = ' '
        matrix_shape = (8,)  # 每行是一个 8 元素的数组
        trajectory_indices = (slice(None), slice(1, 4))  # 提取第 1 到第 3 列 (x, y, z)
    else:
        raise ValueError("Unsupported path_type. Use 'matrix' or 'quaternion'.")

    # 读取轨迹数据
    trajectory = read_poses(file_path, sep=sep, matrix_shape=matrix_shape, trajectory_indices=trajectory_indices)

    # 绘制轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
    # 标注初始点和结束点
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=100, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', s=100, label='End')

    # 打印初始点和结束点的位置
    print("Start Point:", trajectory[0, 0], trajectory[0, 1], trajectory[0, 2])
    print("End Point:", trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2])

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