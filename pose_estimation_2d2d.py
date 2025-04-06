import cv2
import numpy as np

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

def pixel2cam(p, K):
    """
    将二维像素坐标转换为相机坐标系中的归一化坐标。
    
    输入：
    p - 二维像素坐标
    K - 相机内参矩阵
    
    输出：
    cam_coords - 相机坐标系中的归一化二维坐标
    """
    return np.array([(p[0] - K[0, 2]) / K[0, 0], (p[1] - K[1, 2]) / K[1, 1]])

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

def pose_estimation_2d2d(keypoints_1, keypoints_2, matches, K):
    """
    使用2D-2D特征匹配估计相机的运动（旋转矩阵和平移向量）
    """
    # 提取匹配点的坐标
    points1 = np.array([keypoints_1[m[0]] for m in matches])
    points2 = np.array([keypoints_2[m[1]] for m in matches])
    
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

def main(img_path_1, img_path_2, keypoints_1, keypoints_2, matches, K, R1, t1):
    """
    主程序，加载图像，估计相机运动
    """
    # 读取图像
    img_1 = cv2.imread(img_path_1, cv2.IMREAD_COLOR)
    img_2 = cv2.imread(img_path_2, cv2.IMREAD_COLOR)
    
    # 估计相机的运动（旋转矩阵、平移向量和本质矩阵）
    R2, t2, E = pose_estimation_2d2d(keypoints_1, keypoints_2, matches, K)
    
    # 将第二帧的相对位姿转换为绝对位姿
    R2_absolute, t2_absolute = transform_pose(R1, t1, R2, t2)
    print("Second Frame Absolute Pose (Rotation Matrix):\n", R2_absolute)
    print("Second Frame Absolute Pose (Translation Vector):\n", t2_absolute)
    
    q2_absolute = rotation_matrix_to_quaternion(R2_absolute)
    print("Second Frame Absolute Pose (Quaternion):\n", q2_absolute)
    
    # 验证对极约束
    for m in matches:
        pt1 = pixel2cam(keypoints_1[m[0]], K)
        pt2 = pixel2cam(keypoints_2[m[1]], K)
        
        # 将像素点转换为齐次坐标
        y1 = np.array([pt1[0], pt1[1], 1])
        y2 = np.array([pt2[0], pt2[1], 1])
        
        # 计算对极约束
        epipolar_constraint = np.dot(y2.T, np.dot(E, y1))
        print(f"Epipolar constraint = {epipolar_constraint}")
        
if __name__ == '__main__':
    # 输入第一帧的相机姿态：平移向量t1 和四元数q1
    t1 = np.array([-1.269599999999999795e+01, 7.138000000000005230e+00, 5.965000000000003411e+00]).reshape(3, 1) # 第一帧的平移向量
    q1 = np.array([-3.620430825582236500e-01, -2.951450722324931997e-01, 2.362065215861649803e-01, 8.520684666555735642e-01])  # 第一帧的四元数
    
    t2 = np.array([-1.487399999999999523e+01, 8.453000000000002956e+00, 7.268000000000000682e+00]).reshape(3, 1)
    q2 = np.array([-3.810581850187971797e-01, -3.102249397251965468e-01, 2.591547539737096284e-01, 8.315010281987054164e-01])
    
    # 计算第一帧的旋转矩阵
    R1 = quaternion_to_rotation_matrix(q1)
    R2 = quaternion_to_rotation_matrix(q2)
    
    # 这里填充相机内参
    fx = 447.679079249249
    fy = 448.768431364843
    cx = 252.560061902772
    cy = 233.658637132913
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # 你手动输入的特征点坐标 (x, y)
    keypoints_1 = [(374.0, 179.0), (171.0, 351.0), 
                   (170.0, 287.0), (426.0, 74.0), 
                   (421.0, 74.0), (366.0, 415.0), 
                   (256.0, 288.0), (419.0, 339.0)] 
    keypoints_2 = [(357.6009521484375, 201.79965209960938), (177.0801544189453, 355.17535400390625), 
                   (176.5126953125, 296.0967712402344), (398.00860595703125, 119.263671875), 
                   (393.70037841796875, 118.0805892944336), (353.1407775878906, 408.09637451171875),
                   (254.03746032714844, 296.9834899902344), (397.08203125, 344.8715515136719)]
    
    keypoints_3 = [(333.0247802734375, 57.1717529296875), (203.02484130859375, 367.6008605957031), 
                   (203.76791381835938, 323.6740417480469), (358.25732421875, 211.51919555664062), 
                   (355.2914733886719, 210.10787963867188), (331.1205749511719, 404.42340087890625), 
                   (260.1688232421875, 325.3172607421875), [360.3055419921875, 356.3589172363281]]
    
    # 手动输入的匹配点索引对 (keypoints_1, keypoints_2)
    matches = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
    
    # 图片路径（根据实际情况修改）
    img_path_1 = '000000.jpg'
    img_path_2 = '000040.jpg'
    img_path_3 = '000080.jpg'
    
    # 运行主程序
    main(img_path_2, img_path_3, keypoints_2, keypoints_3, matches, K, R2, t2)
