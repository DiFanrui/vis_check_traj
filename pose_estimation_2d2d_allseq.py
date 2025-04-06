import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非图形界面的后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

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

def rotation_matrix_to_quaternion(R):
    """
    将旋转矩阵转换为四元数
    输入：
    R - 旋转矩阵 3x3
    
    输出：
    q - 四元数，格式为 [qx, qy, qz, qw]
    """
    # 确保 R 是一个 3x3 矩阵
    assert R.shape == (3, 3)
    
    # 计算四元数
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    
    return np.array([qx, qy, qz, qw])

def load_pose_point_map(json_path):
    """
    加载 JSON 文件并解析为帧与点的映射。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_poses(txt_path):
    """
    加载 txt 文件并解析为每一帧的位姿。
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    t_list = []
    q_list = []
    R_list = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 8:
            # 解析位姿
            frame, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            t = np.array([tx, ty, tz]).reshape(3, 1)
            q = np.array([qx, qy, qz, qw])
            R = quaternion_to_rotation_matrix(q)
            t_list.append(t)
            q_list.append(q)
            R_list.append(R)
    return t_list, q_list, R_list

def find_matches_between_frames(frame1_points, frame2_points, num_matches=8):
    """
    在两帧之间找到匹配点对。
    
    输入：
    - frame1_points: 第一帧的点 {点ID: [x, y]}
    - frame2_points: 第二帧的点 {点ID: [x, y]}
    - num_matches: 每两帧之间需要的匹配点数量
    
    输出：
    - matches: 匹配点对 [(点ID1, 点ID2), ...]
    """
    matches = []
    # 找到两帧中共有的点 ID
    common_ids = set(frame1_points.keys()) & set(frame2_points.keys())
    for point_id in common_ids:
        matches.append((point_id, point_id))
        if len(matches) == num_matches:
            break
    
    # 如果匹配点不足，基于距离补全
    if len(matches) < num_matches:
        remaining_ids1 = set(frame1_points.keys()) - common_ids
        remaining_ids2 = set(frame2_points.keys()) - common_ids
        
        # 将剩余点转换为列表
        remaining_points1 = [(pid, frame1_points[pid]) for pid in remaining_ids1]
        remaining_points2 = [(pid, frame2_points[pid]) for pid in remaining_ids2]
        
        # 计算距离并匹配最近点
        for pid1, p1 in remaining_points1:
            distances = [(pid2, np.linalg.norm(np.array(p1) - np.array(frame2_points[pid2]))) for pid2 in remaining_ids2]
            distances.sort(key=lambda x: x[1])  # 按距离排序
            if distances:
                pid2, _ = distances[0]
                matches.append((pid1, pid2))
                remaining_ids2.remove(pid2)  # 移除已匹配的点
                if len(matches) == num_matches:
                    break
    
    return matches

def generate_matches_for_sequence(pose_point_map, num_matches=8):
    """
    为整个序列生成匹配点对。
    
    输入：
    - json_path: JSON 文件路径
    - num_matches: 每两帧之间需要的匹配点数量
    
    输出：
    - all_matches: 每两帧的匹配点对 {帧ID: [(点ID1, 点ID2), ...]}
    """
    all_matches = {}
    
    frame_ids = sorted(map(int, pose_point_map.keys()))  # 获取所有帧 ID 并排序
    for i in range(len(frame_ids) - 1):
        frame1_id = frame_ids[i]
        frame2_id = frame_ids[i + 1]
        
        # 提取两帧的点
        frame1_points = {p[0]: p[1] for p in pose_point_map[str(frame1_id)]}
        frame2_points = {p[0]: p[1] for p in pose_point_map[str(frame2_id)]}
        
        # 找到匹配点对
        matches = find_matches_between_frames(frame1_points, frame2_points, num_matches)
        all_matches[(frame1_id, frame2_id)] = matches
    
    return all_matches

def pose_estimation_2d2d(keypoints_1, keypoints_2, K):
    """
    使用2D-2D特征匹配估计相机的运动（旋转矩阵和平移向量）
    输入：
    - keypoints_1: 第一帧的关键点列表
    - keypoints_2: 第二帧的关键点列表
    - K: 相机内参矩阵
    
    输出：
    - R: 旋转矩阵
    - t: 平移向量
    - essential_matrix: 本质矩阵
    """
    # 转换为 numpy 数组
    points1 = np.array(keypoints_1)
    points2 = np.array(keypoints_2)
    
    # 计算基础矩阵
    fundamental_matrix, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT)
    print("Fundamental Matrix:\n", fundamental_matrix)
    
    # 计算本质矩阵
    focal_length = K[0, 0]  # 相机焦距
    principal_point = (K[0, 2], K[1, 2])  # 相机光心
    essential_matrix, _ = cv2.findEssentialMat(points1, points2, focal_length, principal_point)
    print("Essential Matrix:\n", essential_matrix)
    
    # 从本质矩阵中恢复旋转矩阵和平移向量
    _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, focal=focal_length, pp=principal_point)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)
    
    return R, t, essential_matrix

def transform_pose(R1, t1, R2, t2):
    """
    通过相机姿态转换，计算第二帧的绝对位姿。
    输入：
    R1, t1 - 第一帧的旋转矩阵和平移向量
    R2, t2 - 第二帧的相对旋转矩阵和平移向量
    
    输出：
    R2_absolute, t2_absolute - 第二帧的绝对位姿
    """
    # 计算第二帧的绝对旋转矩阵和平移向量
    R2_absolute = np.dot(R1, R2)
    t2_absolute = np.dot(R1, t2) + t1

    return R2_absolute, t2_absolute

if __name__ == '__main__':
    # point, frame 文件路径
    json_path = '/home/data1/difanrui/Project/vis_check_traj/lung_dataset/config_file/pose_point_map.json'
    # poses 文件路径
    txt_path = '/home/data1/difanrui/Project/vis_check_traj/lung_dataset/groundtruth.txt'
    # 输出 JSON 文件路径
    output_json_path = '/home/data1/difanrui/Project/vis_check_traj/lung_dataset/config_file/match_result.json'
    
    pose_point_map = load_pose_point_map(json_path)
    
    # 加载初始位姿
    t, q, R = load_poses(txt_path)
    
    # 每两帧之间匹配点数量
    num_matches = 8
    
    # 生成匹配点对
    all_matches = generate_matches_for_sequence(pose_point_map, num_matches)
    
    # 相机内参
    fx = 447.679079249249
    fy = 448.768431364843
    cx = 252.560061902772
    cy = 233.658637132913
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # 存储结果的字典
    results = {"frames": []}
    
    # 遍历每对帧，计算绝对位姿
    for (frame1_id, frame2_id),matches in all_matches.items():
        # 提取两帧的关键点
        frame1_id_str = str(frame1_id)
        frame2_id_str = str(frame2_id)
        
        # 打印调试信息
        print(f"Processing frames: {frame1_id_str}, {frame2_id_str}")
        
        # 提取两帧的关键点
        if frame1_id_str not in pose_point_map or frame2_id_str not in pose_point_map:
            print(f"Frame {frame1_id_str} or {frame2_id_str} not found in pose point map!")
            continue
        
        frame1_points = {p[0]: p[1] for p in pose_point_map[frame1_id_str]}
        frame2_points = {p[0]: p[1] for p in pose_point_map[frame2_id_str]}
        
        # 根据匹配信息提取关键点
        keypoints_1 = [frame1_points[m[0]] for m in matches if m[0] in frame1_points]
        keypoints_2 = [frame2_points[m[1]] for m in matches if m[1] in frame2_points]
        
        # 检查是否有足够的匹配点
        if len(keypoints_1) < 8 or len(keypoints_2) < 8:
            print(f"Not enough matched points between frames {frame1_id_str} and {frame2_id_str}. Skipping...")
            continue
        
        # 估计相对位姿
        R_relative, t_relative, E = pose_estimation_2d2d(keypoints_1, keypoints_2, K)
        
        # 获取上一帧的绝对位姿
        R1 = R[frame1_id]
        t1 = t[frame1_id]
        
        # 将第二帧的相对位姿转换为绝对位姿
        R_absolute, t_absolute = transform_pose(R1, t1, R_relative, t_relative)
        q_absolute = rotation_matrix_to_quaternion(R_absolute)
        
        # 保存计算的绝对位姿
        R.append(R_absolute)
        t.append(t_absolute)
        q.append(q_absolute)

        # 保存到结果字典
        results["frames"].append({
            "frame_id": frame2_id,
            "input_pose": {
                "rotation_matrix": R1.tolist(),
                "translation_vector": t1.flatten().tolist(),
                "quaternion": q[frame1_id].tolist()
            },
            "calculated_pose": {
                "rotation_matrix": R_absolute.tolist(),
                "translation_vector": t_absolute.flatten().tolist(),
                "quaternion": q_absolute.tolist()
            }
        })
    
    # 将结果保存到 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_json_path}")
