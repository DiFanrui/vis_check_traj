import numpy as np
import json
from pathlib import Path

def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def make_transform_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def read_poses(file_path, sep=' ', matrix_shape=(8,), trajectory_indices=(slice(None), slice(1, 4)), orientation_indices=(slice(None), slice(4, 8))):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            pose = np.fromstring(line, sep=sep).reshape(matrix_shape)
            poses.append(pose)

    poses = np.array(poses)
    trajectory = poses[trajectory_indices]
    orientations = poses[orientation_indices] if orientation_indices is not None else None
    return trajectory, orientations

def split_frames(frames):
    train_frames, test_frames, val_frames = [], [], []
    for i, frame in enumerate(frames):
        mod = i % 10
        if mod < 7:
            train_frames.append(frame)
        elif mod < 9:
            test_frames.append(frame)
        else:
            val_frames.append(frame)
    return train_frames, test_frames, val_frames

def convert_to_blender_format(pose_file_path, image_dir, image_extension=".png", image_width=480):
    fx = 447.679079249249
    cx = 252.560061902772
    camera_angle_x = 2 * np.arctan(image_width / (2 * fx))

    frames = []
    trajectory, orientations = read_poses(pose_file_path)

    for idx, (t, q) in enumerate(zip(trajectory, orientations)):
        R = quaternion_to_rotation_matrix(q)
        T = make_transform_matrix(R, t)
        T[:3, 1:3] *= -1

        frame = {
            "file_path": f"{image_dir}/{idx:08d}",
            "transform_matrix": T.tolist()
        }
        frames.append(frame)

    train_frames, test_frames, val_frames = split_frames(frames)

    def save_json(frames, out_name):
        with open(out_name, "w") as f:
            json.dump({"camera_angle_x": camera_angle_x, "frames": frames}, f, indent=4)

    base = Path(pose_file_path).parent
    save_json(train_frames, base / "transforms_train.json")
    save_json(test_frames, base / "transforms_test.json")
    save_json(val_frames, base / "transforms_val.json")

    print("✅ transforms_train.json / test / val 已生成完毕！")

# 示例调用
convert_to_blender_format(
    pose_file_path="/home/data1/difanrui/Project/vis_check_traj/blender_type/groundtruth.txt",
    image_dir="/home/data1/difanrui/Project/vis_check_traj/blender_type/images",
    image_extension=".jpg",
    image_width=480
) 
