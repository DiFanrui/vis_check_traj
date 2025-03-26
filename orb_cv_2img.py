import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

def feature_extraction(img1_path, img2_path):
    # 读取图像
    img_1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img_2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    assert img_1 is not None and img_2 is not None, "无法读取图像，请检查路径是否正确！"

    # 将图像从 BGR 转换为 RGB（matplotlib 使用 RGB 格式）
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

    # 初始化 ORB 检测器和匹配器
    detector = cv2.ORB_create(nfeatures=2000)  # 增加特征点数量
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 提取特征点和描述子
    keypoints_1, descriptors_1 = detector.detectAndCompute(img_1, None)
    keypoints_2, descriptors_2 = detector.detectAndCompute(img_2, None)

    # 特征匹配
    matches = matcher.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)  # 按距离排序

    # 计算最小距离和最大距离
    min_dist = matches[0].distance
    max_dist = matches[-1].distance
    print("-- 最大距离 : {:.2f}".format(max_dist))
    print("-- 最小距离 : {:.2f}".format(min_dist))

    # 过滤好的匹配点
    good_matches = [m for m in matches if m.distance <= max(2 * min_dist, 15.0)]  # 更严格的过滤条件

    # 绘制匹配结果
    img_goodmatch = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, None)

    # 使用 matplotlib 显示匹配结果
    plt.figure(figsize=(12, 6))
    plt.imshow(img_goodmatch)
    plt.title("matched points")
    plt.axis("off")  # 隐藏坐标轴
    output_path = "verify_traingulation/matches.png"  # 保存路径
    plt.savefig(output_path, bbox_inches="tight")  # 保存图像
    plt.show()
    print(f"匹配结果已保存到 {output_path}")

    # 保存特征点到 JSON 文件（格式化）
    keypoints_dict = {
        'image1': {i: [kp.pt[0],  kp.pt[1]] for i, kp in enumerate(keypoints_1)},
        'image2': {i: [kp.pt[0],  kp.pt[1]] for i, kp in enumerate(keypoints_2)}
    }
    with open('verify_traingulation/keypoints.json', 'w') as f:
        json.dump(keypoints_dict, f, indent=4)  # 使用缩进格式化 JSON 文件

    # 保存匹配点到 JSON 文件
    matches_dict = {
        'matches': [
            {
                'points_in_img1_idx': m.queryIdx,
                'image1_point': [keypoints_1[m.queryIdx].pt[0], keypoints_1[m.queryIdx].pt[1]],
                'points_in_img2_idx': m.trainIdx,
                'image2_point': [keypoints_2[m.trainIdx].pt[0], keypoints_2[m.trainIdx].pt[1]],
                'distance': m.distance
            }
            for m in good_matches
        ]
    }
    with open('verify_traingulation/matches.json', 'w') as f:
        json.dump(matches_dict, f, indent=4)  # 使用缩进格式化 JSON 文件

if __name__ == "__main__":
    img1_path = "verify_traingulation/1.png"
    img2_path = "verify_traingulation/2.png"
    feature_extraction(img1_path, img2_path)