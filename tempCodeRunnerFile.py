import numpy as np
import cv2
from matplotlib import pyplot as plt

# 讀取左右相機影像
left_image = cv2.imread(r'D:\Lab\111-2\computer_vision_and_image_measurement\test_crack_image_L.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(r'D:\Lab\111-2\computer_vision_and_image_measurement\test_crack_image_R.png', cv2.IMREAD_GRAYSCALE)

# 創建特徵檢測器
orb = cv2.ORB_create()

# 在左右影像中檢測特徵點和描述符
left_keypoints, left_descriptors = orb.detectAndCompute(left_image, None)
right_keypoints, right_descriptors = orb.detectAndCompute(right_image, None)

# 創建特徵匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 進行特徵匹配
matches = bf.match(left_descriptors, right_descriptors)

# 根據特徵點匹配的距離進行排序
matches = sorted(matches, key=lambda x: x.distance)

# 選擇前n個最佳匹配
n = 1000
best_matches = matches[:n]

# 提取最佳匹配的特徵點在左右影像中的位置
left_points = np.float32([left_keypoints[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
right_points = np.float32([right_keypoints[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# 进行初步筛选
min_distance = 10000
max_distance = 0
for x in best_matches:
    if x.distance < min_distance:
        min_distance = x.distance
    if x.distance > max_distance:
        max_distance = x.distance
print('Min_Dis：%f' % min_distance)
print('Max_Dis：%f' % max_distance)

filtered_matches = []
for x in best_matches:
    if x.distance <= max(2 * min_distance, 30):
        filtered_matches.append(x)

print('Match_Num：%d' % len(filtered_matches))

# 繪製篩選後的匹配結果
outimage = cv2.drawMatches(left_image, left_keypoints, right_image, right_keypoints, filtered_matches, outImg=None)
plt.imshow(outimage[:,:,::-1])
plt.show()
