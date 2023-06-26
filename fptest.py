import numpy as np

def findNearestPointsFromFeature(image_points, points_3d_list):
    for i in range(len(points_3d_list)):
        distance = np.linalg.norm(image_points - points_3d_list[i][0])
        print(distance)

points_3d_list = [np.array(), np.array()]
image_points = np.array([[100, 100]], dtype=np.float32)
nearest_leftimagePoint = findNearestPointsFromFeature(image_points, points_3d_list)