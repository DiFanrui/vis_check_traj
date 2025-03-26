import cv2
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
    
    # 计算旋转矩阵的元素
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R

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

def triangulate_points(K, R1, t1, R2, t2, pt1_2d, pt2_2d):
    """
    使用三角测量方法从两张视图的二维点计算三维点。
    
    输入：
    K - 相机内参矩阵
    R1, t1 - 视图1的旋转矩阵和平移向量
    R2, t2 - 视图2的旋转矩阵和平移向量
    pt1_2d, pt2_2d - 视图1和视图2中的二维点坐标
    
    输出：
    pts_3d - 计算得到的三维点坐标
    """
    # 将像素坐标转换为相机坐标
    pt1_cam = pixel2cam(pt1_2d, K)
    pt2_cam = pixel2cam(pt2_2d, K)
    
    # 计算投影矩阵
    T1 = np.hstack((R1, t1.reshape(3, 1)))  # 视图1的投影矩阵
    T2 = np.hstack((R2, t2.reshape(3, 1)))  # 视图2的投影矩阵
    
    # 三角化过程
    pts_1 = pt1_cam.reshape(2, 1)
    pts_2 = pt2_cam.reshape(2, 1)
    
    # 使用OpenCV进行三角化
    pts_4d_hom = cv2.triangulatePoints(T1, T2, pts_1, pts_2)
    
    # 转换为非齐次坐标
    pts_3d = pts_4d_hom / pts_4d_hom[3]  # 齐次坐标归一化
    return pts_3d[:3]  # 返回三维点坐标 (X, Y, Z)


# 这里填充相机内参和外参
# 相机内参 (例如：fx, fy, cx, cy)
fx = 520.9
fy = 521.0
cx = 325.1
cy = 249.7
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


# 第一张
q1 = np.array([-0.5721, 0.6521, -0.3565, 0.3469])  
R1 = quaternion_to_rotation_matrix(q1)  
t1 = np.array([0.1163, -1.1498, 1.4015]) 

# 第二张
q2 = np.array([-0.6085, 0.6261, -0.3409, 0.3486]) 
R2 = quaternion_to_rotation_matrix(q2) 
t2 = np.array([0.0797, -1.0447, 1.4561])  

# 第三张
q3 = np.array([-0.6180, 0.6240, -0.3245, 0.3513]) 
R3 = quaternion_to_rotation_matrix(q2) 
t3 = np.array([0.0726, -0.9638, 1.4661])  

# 视图1和视图2中的二维点坐标 p1(x1, y1) 和 p2(x2, y2)
pt1_2d = np.array([351.1518249511719, 300.9872741699219])  # 第0帧
pt2_2d = np.array([426.3986511230469, 297.4040832519531])  # 第40帧
pt3_2d = np.array([480.1463623046875, 286.654541015625])  # 第80帧

# 计算三维点并验证
reconstructed_3d_point1 = triangulate_points(K, R1, t1, R2, t2, pt1_2d, pt2_2d)
reconstructed_3d_point2 = triangulate_points(K, R1, t1, R3, t3, pt1_2d, pt3_2d)
reconstructed_3d_point3 = triangulate_points(K, R2, t2, R3, t3, pt2_2d, pt3_2d)

# 打印三维点
print("第0帧和第8帧三角测量得到的三维点坐标为: ", reconstructed_3d_point1)
print("第0帧和第16帧三角测量得到的三维点坐标为: ", reconstructed_3d_point2)
print("第8帧和第16帧三角测量得到的三维点坐标为: ", reconstructed_3d_point3)

# 使用Open3D可视化三维点、相机位置和预测三维点的位置
def visualize_points_and_cameras(reconstructed_3d_points, camera_positions, camera_orientations):
    # 创建点云对象
    point_cloud = o3d.geometry.PointCloud()
    points = reconstructed_3d_points
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 不同颜色表示不同的三维点
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 创建相机框架
    camera_frames = []
    for position, orientation in zip(camera_positions, camera_orientations):
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        camera_frame.rotate(orientation, center=(0, 0, 0))
        camera_frame.translate(position)
        camera_frames.append(camera_frame)

    # 可视化
    o3d.visualization.draw_geometries([point_cloud] + camera_frames)

# 相机位置和朝向
camera_positions = [t1, t2, t3]
camera_orientations = [R1, R2, R3]

# 可视化
visualize_points_and_cameras([reconstructed_3d_point1, reconstructed_3d_point2, reconstructed_3d_point3], camera_positions, camera_orientations)

# 绘制极线和二维点位置
def draw_epilines(img1, img2, lines, pts1, pts2):
    ''' img1 - 图像1
        img2 - 图像2
        lines - 极线
        pts1 - 图像1中的点
        pts2 - 图像2中的点
    '''
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

# 读取图像
img1 = np.zeros((480, 480, 3), dtype=np.uint8)
img2 = np.zeros((480, 480, 3), dtype=np.uint8)

# 计算基础矩阵
pts1 = np.array([pt1_2d])
pts2 = np.array([pt2_2d])
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

# 计算极线
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)

# 绘制极线
img1, img2 = draw_epilines(img1, img2, lines1, pts1, pts2)
img2, img1 = draw_epilines(img2, img1, lines2, pts2, pts1)

# 显示图像
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 使用Open3D可视化三维点、相机位置和预测三维点的位置
# def visualize_points_and_cameras(reconstructed_3d_point1, reconstructed_3d_point2,reconstructed_3d_point3, camera_positions, camera_orientations):
#     # 创建点云对象
#     point_cloud = o3d.geometry.PointCloud()
#     points = [reconstructed_3d_point1, reconstructed_3d_point2,reconstructed_3d_point3]
#     colors = [[1, 0, 0], [0, 1, 0]]  # 红色为真实三维点，绿色为重建的三维点
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     point_cloud.colors = o3d.utility.Vector3dVector(colors)

#     # 创建相机框架
#     camera_frames = []
#     for position, orientation in zip(camera_positions, camera_orientations):
#         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
#         camera_frame.rotate(orientation, center=(0, 0, 0))
#         camera_frame.translate(position)
#         camera_frames.append(camera_frame)

#     # 可视化
#     o3d.visualization.draw_geometries([point_cloud] + camera_frames)

# # 相机位置和朝向
# camera_positions = [t1, t2, t3]
# camera_orientations = [R1, R2, R3]

# # 可视化
# visualize_points_and_cameras(reconstructed_3d_point1, reconstructed_3d_point2,reconstructed_3d_point3, camera_positions, camera_orientations)