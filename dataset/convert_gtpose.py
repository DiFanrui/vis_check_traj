import numpy as np

# 定义四元数乘法函数
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2   # z
    ])

# 定义四元数转换函数
def transform_quaternion(q):
    # 绕 y 轴旋转 180 度的四元数
    q_180 = np.array([0, 1, 0, 0])
    # 将原四元数与绕 y 轴旋转 180 度的四元数相乘
    q_transformed = quaternion_multiply(q_180, q)
    return q_transformed

# 读取文件并转换轨迹
def transform_trajectory(file_path, output_file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 打开输出文件
    with open(output_file_path, 'w') as output_file:
        # 写入文件头
        output_file.write("# Transformed ground truth trajectory\n")
        output_file.write("# file: stable_seq_005_part_1_dif_1_transformed\n")
        output_file.write("# timestamp tx ty tz qx qy qz qw\n")
        
        # 遍历每一行
        for line in lines:
            if line.startswith("#"):
                # 跳过注释行
                continue
            # 解析数据
            parts = line.split()
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            # 转换位置
            tx_transformed = tx
            ty_transformed = -ty
            tz_transformed = tz
            
            # 转换四元数
            q = np.array([qw, qx, qy, qz])
            q_transformed = transform_quaternion(q)
            qw_transformed, qx_transformed, qy_transformed, qz_transformed = q_transformed
            
            # 写入转换后的数据
            output_file.write(f"{timestamp:.12e} {tx_transformed:.12e} {ty_transformed:.12e} {tz_transformed:.12e} "
                              f"{qx_transformed:.12e} {qy_transformed:.12e} {qz_transformed:.12e} {qw_transformed:.12e}\n")

# 输入文件路径和输出文件路径
input_file_path = "/home/data1/difanrui/Project/vis_check_traj/dataset/groundtruth.txt"
output_file_path = "/home/data1/difanrui/Project/vis_check_traj/dataset/groundtruth_converted.txt"

# 调用函数进行转换
transform_trajectory(input_file_path, output_file_path)
print(f"Transformation complete. Transformed trajectory saved to {output_file_path}")