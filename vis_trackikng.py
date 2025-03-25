import numpy as np
import matplotlib.pyplot as plt

def read_invmatrix_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                pose = np.fromstring(line, sep=',').reshape(4, 4)
                poses.append(pose)
    return np.array(poses)

def read_quaternion_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                pose = np.fromstring(line, sep=' ')
                poses.append(pose)
    return np.array(poses)

def read_invmatrix_trajectory(poses):
    # 提取轨迹点
    trajectory = poses[:, 3, :]
    return trajectory

def read_quaternion_trajectory(poses):
    # 提取轨迹点
    trajectory = poses[:, 1:4]
    return trajectory
    
    
def plot_trajectory(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    trajectory = read_quaternion_trajectory(poses)

    # 绘制轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
    # 标注初始点和结束点
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=100, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', s=100, label='End')

    # 标注初始点和结束点的位置
    print("start:",trajectory[0, 0], trajectory[0, 1], trajectory[0, 2])
    print("end:",trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2])



    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Visualization')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    # file_path = 'D:\Project\mydataset\IROS_bronchoscopy_dataset\tum_stable_seq_005_part_1_dif_1\pose_inv.txt'
    file_path = "dataset/groundtruth_converted.txt"
    poses = read_quaternion_poses(file_path)
    plot_trajectory(poses)