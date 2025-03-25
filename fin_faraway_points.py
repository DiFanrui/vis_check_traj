import json
import numpy as np

def load_points(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def find_far_points(points_data, threshold):
    far_points = {}
    for point_id, point_list in points_data.items():
        point_xyz = point_list[0]  # 三维坐标
        # 读取点的三维坐标模长
        # distance = np.linalg.norm(point_xyz)
        # if distance > threshold:
        #     far_points[point_id] = point_list
        if any(abs(coord) > threshold for coord in point_xyz):
            far_points[point_id] = point_list
    return far_points

def save_far_points(far_points, output_file):
    with open(output_file, 'w') as f:
        json.dump(far_points, f, indent=4)

# 加载点云数据
json_file = "/home/data1/difanrui/Project/vis_check_traj/points.json"
points_data = load_points(json_file)

# 找出距离原点超过某个阈值的点
threshold = 100  # 设定阈值
far_points = find_far_points(points_data, threshold)

# 保存结果到新的 JSON 文件
output_file = "/home/data1/difanrui/Project/vis_check_traj/far_points.json"
save_far_points(far_points, output_file)

print(f"找到 {len(far_points)} 个距离超过 {threshold} 的点，并保存到 {output_file}")