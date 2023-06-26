import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from numba import njit

# 讀取左右相機影像
#left_image = cv2.imread(r'D:\Lab\111-2\computer_vision_and_image_measurement\test_crack_image_L.png')
#right_image = cv2.imread(r'D:\Lab\111-2\computer_vision_and_image_measurement\test_crack_image_R.png')

left_image = cv2.imread(r'D:\Lab\111-2\computer_vision_and_image_measurement\test_crack_mask_L_binary.png')
right_image = cv2.imread(r'D:\Lab\111-2\computer_vision_and_image_measurement\test_crack_mask_R_binary.png')

# 讀取左右相機 Mask 影像
left_mask = cv2.imread(r'D:\Lab\111-2\computer_vision_and_image_measurement\test_crack_mask_L_binary.png')
right_mask = cv2.imread(r'D:\Lab\111-2\computer_vision_and_image_measurement\test_crack_mask_R_binary.png')

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
n = 100
best_matches = matches[:n]

# 提取最佳匹配的特徵點在左右影像中的位置
left_points = np.float32([left_keypoints[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
right_points = np.float32([right_keypoints[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

min_distance = 10000
max_distance = 0
for x in best_matches:
    if x.distance < min_distance:
        min_distance = x.distance
    if x.distance > max_distance:
        max_distance = x.distance
#print('Min_Dis：%f' % min_distance)
#print('Max_Dis：%f' % max_distance)

filtered_matches = []
for x in best_matches:
    if x.distance <= max(2 * min_distance, 30):
        filtered_matches.append(x)

#print('Match_Num：%d' % len(filtered_matches))

# 繪製篩選後的匹配結果
outimage = cv2.drawMatches(left_image, left_keypoints, right_image, right_keypoints, filtered_matches, outImg=None)
#plt.imshow(outimage[:,:,::-1])
#plt.show()

'''
# CAM L ##############################################################
K
<Matrix 3x3 (622.2222,   0.0000, 224.0000)
            (  0.0000, 622.2222, 224.0000)
            (  0.0000,   0.0000,   1.0000)>
RT
<Matrix 3x4 (0.9806, -0.1961,  0.0000, -0.0000)
            (0.0000,  0.0000, -1.0000,  0.0000)
            (0.1961,  0.9806,  0.0000,  7.6485)>
            
# CAM R ##############################################################
K
<Matrix 3x3 (622.2222,   0.0000, 224.0000)
            (  0.0000, 622.2222, 224.0000)
            (  0.0000,   0.0000,   1.0000)>
RT
<Matrix 3x4 ( 0.9806, 0.1961,  0.0000, 0.0000)
            ( 0.0000, 0.0000, -1.0000, 0.0000)
            (-0.1961, 0.9806,  0.0000, 7.6485)>
'''
# 定義左右相機的內部參數和外部參數
# 以下參數僅供示範，請根據實際情況填入正確的值
focal_length1 = 622.2222
principal_point1 = (224,224)
focal_length2 = 622.2222
principal_point2 = (224,224)

camera_matrix1 = np.array([[focal_length1, 0, principal_point1[0]],
                          [0, focal_length1, principal_point1[1]],
                          [0, 0, 1]])
camera_matrix2 = np.array([[focal_length2, 0, principal_point2[0]],
                          [0, focal_length2, principal_point2[1]],
                          [0, 0, 1]])

# 假設你已經有兩台相機各自的旋轉矩陣和平移向量
rotation_matrix1 = np.array([[0.9806, -0.1961, 0.0000],
                            [0.0000, 0.0000, -1.0000],
                            [0.1961, 0.9806, 0.0000]])
translation_vector1 = np.array([[-0.0000],
                               [0.0000],
                               [7.6485]])

rotation_matrix2 = np.array([[0.9806, 0.1961,  0.0000],
                            [0.0000, 0.0000, -1.0000],
                            [-0.1961, 0.9806, 0.0000]])
translation_vector2 = np.array([[0.0000],
                               [0.0000],
                               [7.6485]])

# 計算外部參數矩陣 [R | t]
RT1 = np.hstack((rotation_matrix1, translation_vector1))
RT2 = np.hstack((rotation_matrix2, translation_vector2))

# 計算投影矩陣 P
P1 = np.dot(camera_matrix1, RT1)
P2 = np.dot(camera_matrix2, RT2)

# 進行三角測量
points_4d_homogeneous = cv2.triangulatePoints(P1, P2, left_points, right_points)

# 將齊次座標轉換為三維座標形式
points_3d = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T)

points_3d_list = []
# 印出三維座標
# points_3d_list = [np.array(2D_coord), np.array(3D_coord)]
for i, point in enumerate(points_3d):
    #print(f"Point {i+1}: {point.flatten()}")
    points_3d_list.append([left_points[i], point.flatten()])
#print(points_3d_list)
# 取Y向的值

#print(max_Y)
#print(min_Y)
'''
def Cal3DCoordFrom2DCoord(x,y):
    point_P_homogeneous = np.array([[x], [y], [1]])
    point_P_3d_homogeneous = np.dot(np.linalg.inv(camera_matrix1), point_P_homogeneous)
    point_P_3d = point_P_3d_homogeneous[:3]
    point_P_3d = point_P_3d.flatten()
    #print(f"Point P in 3D: {point_P_3d}")
    return point_P_3d

Cal3DCoordFrom2DCoord(100,100)
'''
# 現在已知裂縫平面大概位置，可以定出範圍，以最大的深度及最小深度設定，或許可以各往外加一點
# 用左相機 mask 的影像取出裂縫的輪廓
# 再用右相機 mask 的影像取出裂縫的輪廓
# 可以利用限制深度範圍的方式計算所有裂縫輪廓的三維點位
# initial guess 用前面做的已知二維轉三維猜，找最接近的

def cal_3D_points(image_points, rvec, tvec, camera_matrix, dist_coeffs, nearest_leftimagePoint):
    # 定義投影誤差函式
    def reprojection_error(params, rvec, tvec, camera_matrix, dist_coeffs, image_points):
        # 提取優化參數
        x, y, z = params

        # 將三維點位轉換維相機座標系下的座標
        camera_point = np.array([x, y, z])

        # 進行投影變換，將三維點位投影到圖向平面
        image_point_projected, _ = cv2.projectPoints(camera_point, rvec, tvec, camera_matrix, dist_coeffs)

        # 計算投影誤差
        error = np.linalg.norm(image_points - image_point_projected)

        return error
    
    x = image_points[0][0]
    y = image_points[0][1]
    # 初始化優化參數
    initial_params = np.array([x, 0, y], dtype=np.float32)
    # Y bounds
    points_3d_Y = points_3d[:, 0, 1]
    # 算最大值及最小值
    max_Y = np.max(points_3d_Y)
    min_Y = np.min(points_3d_Y)
    # 設定參數上下限(主要要限制Y向變化)
    lower_bounds = [-1000, 1.5 * min_Y , -1000] 
    upper_bounds = [1000, 1.5 * max_Y, 1000] 
    bounds = (lower_bounds, upper_bounds)

    # 用 least square 進行優化
    result = least_squares(reprojection_error, initial_params, args=(rvec1, tvec1, camera_matrix, dist_coeffs, image_points), bounds=bounds)

    # 取得優化後的三維點位
    optimized_params = result.x
    x_opt, y_opt, z_opt = optimized_params

    #print(np.array([x_opt, y_opt, z_opt], dtype=float))
    objpointCalFromImage = np.array([x_opt, y_opt, z_opt], dtype=float)
    return objpointCalFromImage
    
@njit
# Find crack pixel from L_camera segmentation image
def findcrackpixel(seg_img):
    crack_pixel = []
    for i in range(seg_img.shape[0]):
        for j in range(seg_img.shape[1]):
            if np.all(seg_img[i, j] == np.array([255, 255, 255])):
                crack_pixel.append([i, j])
    return crack_pixel

crack_pixel = findcrackpixel(left_mask)
'''
print(crack_pixel)

left_mask_co = left_mask.copy()
for pixel in crack_pixel:
    left_mask_co[int(pixel[0]), int(pixel[1])] = (0, 0, 255)
cv2.imshow("crack_pixel", left_mask_co)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# image_points = np.array([[p[1], p[0]]], dtype=np.float32)
# points_3d_list = [np.array(2D_coord), np.array(3D_coord)]
def findNearestPointsFromFeature(image_points, points_3d_list):
    min_distance = 1e15
    min_distance_index = -1

    for i in range(len(points_3d_list)):
        distance = np.linalg.norm(image_points - points_3d_list[i][0])
        #print(distance)

        if distance < min_distance:
            min_distance = distance
            min_distance_index = i

    if min_distance_index != -1:
        min_distance_point_2d = points_3d_list[min_distance_index][0]
        min_distance_point_3d = points_3d_list[min_distance_index][1]

        minDisPointList = [min_distance, min_distance_point_2d, min_distance_point_3d]
        #print(minDisPointList)
    return minDisPointList


# 已知的相機內外參及畸變參數
camera_matrix = camera_matrix1
rvec1 = rotation_matrix1
tvec1 = translation_vector1
dist_coeffs = np.zeros(5, dtype=float)

# 已知的二維點位
crack3Dinfo = []
for i, p in enumerate(crack_pixel):
    image_points = np.array([[p[1], p[0]]], dtype=np.float32)
    nearest_leftimagePoint = findNearestPointsFromFeature(image_points, points_3d_list)
    print(nearest_leftimagePoint)
    print(type(nearest_leftimagePoint))
    objectpoints = cal_3D_points(image_points, rvec1, tvec1, camera_matrix, dist_coeffs, nearest_leftimagePoint)
    crack3Dinfo.append(objectpoints)
    print("Point "+str(i+1)+" 3D Coordinates : ", objectpoints)
print(crack3Dinfo)

# 建立三維圖形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 迴圈畫點(3D)
for point in crack3Dinfo:
    x, y, z = point
    ax.plot([x], [y], [z], marker='o', color='r')

# 設定坐標軸
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 繪圖
plt.show()