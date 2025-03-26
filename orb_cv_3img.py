import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

def feature_extraction_three_images(img1_path, img2_path, img3_path):
    # 读取图像
    img_1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img_2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    img_3 = cv2.imread(img3_path, cv2.IMREAD_COLOR)
    assert img_1 is not None and img_2 is not None and img_3 is not None, "无法读取图像，请检查路径是否正确！"

    # 初始化 ORB 检测器和匹配器
    detector = cv2.ORB_create(nfeatures=2000)  # 增加特征点数量
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 提取特征点和描述子
    keypoints_1, descriptors_1 = detector.detectAndCompute(img_1, None)
    keypoints_2, descriptors_2 = detector.detectAndCompute(img_2, None)
    keypoints_3, descriptors_3 = detector.detectAndCompute(img_3, None)

    # 图像 1 和图像 2 的匹配
    matches_12 = matcher.match(descriptors_1, descriptors_2)
    matches_12 = sorted(matches_12, key=lambda x: x.distance)

    # 图像 2 和图像 3 的匹配
    matches_23 = matcher.match(descriptors_2, descriptors_3)
    matches_23 = sorted(matches_23, key=lambda x: x.distance)

    # 过滤好的匹配点
    good_matches_12 = [m for m in matches_12 if m.distance <= max(2 * matches_12[0].distance, 15.0)]
    good_matches_23 = [m for m in matches_23 if m.distance <= max(2 * matches_23[0].distance, 15.0)]

    # 合并匹配关系
    matches_13 = []
    for m12 in good_matches_12:
        for m23 in good_matches_23:
            if m12.trainIdx == m23.queryIdx:  # 图像 2 的特征点作为桥梁
                matches_13.append({
                    'points_in_img1_idx': m12.queryIdx,
                    'image1_point': [keypoints_1[m12.queryIdx].pt[0], keypoints_1[m12.queryIdx].pt[1]],
                    'points_in_img2_idx': m12.trainIdx,
                    'image2_point': [keypoints_2[m12.trainIdx].pt[0], keypoints_2[m12.trainIdx].pt[1]],
                    'points_in_img3_idx': m23.trainIdx,
                    'image3_point': [keypoints_3[m23.trainIdx].pt[0], keypoints_3[m23.trainIdx].pt[1]],
                    'distance_12': m12.distance,
                    'distance_23': m23.distance
                })

    # 保存匹配点到 JSON 文件
    with open('verify_traingulation/matches.json', 'w') as f:
        json.dump(matches_13, f, indent=4)

    print(f"三张图像的匹配关系已保存到 verify_traingulation/matches_13.json")

    # 可视化匹配结果
    visualize_matches(img_1, img_2, img_3, keypoints_1, keypoints_2, keypoints_3, matches_13)

def visualize_matches(img_1, img_2, img_3, keypoints_1, keypoints_2, keypoints_3, matches_13):
    """
    使用 matplotlib 可视化三张图像的匹配关系
    """
    # 拼接三张图像
    height = max(img_1.shape[0], img_2.shape[0], img_3.shape[0])
    width = img_1.shape[1] + img_2.shape[1] + img_3.shape[1]
    combined_img = np.zeros((height, width, 3), dtype=np.uint8)
    combined_img[:img_1.shape[0], :img_1.shape[1], :] = img_1
    combined_img[:img_2.shape[0], img_1.shape[1]:img_1.shape[1] + img_2.shape[1], :] = img_2
    combined_img[:img_3.shape[0], img_1.shape[1] + img_2.shape[1]:, :] = img_3

    # 绘制匹配关系
    plt.figure(figsize=(20, 10))
    plt.imshow(combined_img)
    for match in matches_13:
        # 图像 1 的点
        x1, y1 = match['image1_point']
        # 图像 2 的点
        x2, y2 = match['image2_point']
        x2 += img_1.shape[1]  # 调整 x 坐标
        # 图像 3 的点
        x3, y3 = match['image3_point']
        x3 += img_1.shape[1] + img_2.shape[1]  # 调整 x 坐标

        # 绘制连线
        plt.plot([x1, x2], [y1, y2], 'r', linewidth=0.5)
        plt.plot([x2, x3], [y2, y3], 'b', linewidth=0.5)

        # 绘制点
        plt.scatter([x1, x2, x3], [y1, y2, y3], c='yellow', s=10)

    plt.axis('off')
    plt.title("matched points")
    output_path = "verify_traingulation/matches.png"  # 保存路径
    plt.savefig(output_path, bbox_inches="tight")  # 保存图像
    plt.show()
    print(f"匹配结果已保存到 {output_path}")

if __name__ == "__main__":
    img1_path = "verify_traingulation/1.png"
    img2_path = "verify_traingulation/2.png"
    img3_path = "verify_traingulation/3.png"
    feature_extraction_three_images(img1_path, img2_path, img3_path)