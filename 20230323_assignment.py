"""
Assignment Week03 :
Define a class that has at least two functions.
func 1 :
    1. Image : 800 x 600 pixels.
    2. Angles of each triangle : between 30 to 90 degrees.
    3. Area of each triangle : 900 to 1100 pixels.
    4. Triangles do not overlap.
    5. Every vertex must be inside the image.
func 2 : Every triangle rotates slowly by 2 pi. And display the animation.
It'll be like :

class MyClass:
    def __init__():
        v = np.zeros((8,2), dtype=float)
    def plot():
        ...
    def rotate():
        ...
    def close():
        ...
    
a = MyClass()
a.plot()
a.rotate()
a.close()
""" 

import cv2
import numpy as np
import random
import math

class MyClass:
    def __init__(self, img_h, img_w, num_of_triangles, max_area, min_area, max_degree, min_degree):
        self.imgsize = (img_h, img_w, 3)
        self.img = np.zeros(self.imgsize,dtype=np.uint8)

        # generate triangle vertex
        self.num_of_triangles = num_of_triangles
        # conditions
        self.max_area = max_area
        self.min_area = min_area
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.vertex_list = []
        for i in range(self.num_of_triangles):
            while True:
                p1 = (random.randint(0 + 50, self.imgsize[1] - 50), random.randint(0 + 50, self.imgsize[0] - 50))
                p2 = (random.randint(0 + 50, self.imgsize[1] - 50), random.randint(0 + 50, self.imgsize[0] - 50))
                p3 = (random.randint(0 + 50, self.imgsize[1] - 50), random.randint(0 + 50, self.imgsize[0] - 50))
                triangle_pts = (p1, p2, p3)
                # check area
                check_area = self.calculate_triangle_area(triangle_pts)
                if check_area <= self.max_area and check_area >= self.min_area:
                    # check degree
                    check_degree = self.cal_degree(triangle_pts)
                    a = check_degree[0]
                    b = check_degree[1]
                    c = check_degree[2]
                    if a <= self.max_degree and a >= self.min_degree and b <= self.max_degree and b >= self.min_degree and c <= self.max_degree and c >= self.min_degree:
                        if len(self.vertex_list) == 0:
                            self.vertex_list.append(triangle_pts)
                            break
                        else:
                            # check overlap
                            overlap = False
                            for last_triangle_pts in self.vertex_list:
                                overlap = self.check_overlap(last_triangle_pts, triangle_pts)
                                if overlap:
                                    break
                            if overlap == False:
                                self.vertex_list.append(triangle_pts)
                                break
   
        
    def calculate_triangle_area(self, triangle_pts):
        # triangle_pts will be : (p1, p2, p3), p1 = (p1x, p1y), ...
        (point1, point2, point3) = triangle_pts
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)
        # Calculate the sides of the triangle
        a = np.linalg.norm(point1 - point2)
        b = np.linalg.norm(point2 - point3)
        c = np.linalg.norm(point3 - point1)
        # Check if the sides form a valid triangle
        if a + b <= c or a + c <= b or b + c <= a:
            return np.nan
        # Use Heron's formula to calculate the area
        s = (a + b + c) / 2
        if (s - a) <= 0 or (s - b) <= 0 or (s - c) <= 0:
            return np.nan
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        
        return area  

    def cal_degree(self, triangle_pts):
        # triangle_pts will be : (p1, p2, p3), p1 = (p1y, p1x), ...
        (point1, point2, point3) = triangle_pts
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)
        # Calculate the sides of the triangle
        a = np.linalg.norm(point1 - point2)
        b = np.linalg.norm(point2 - point3)
        c = np.linalg.norm(point3 - point1)
        # Calculate the angles of the triangle
        cosA = (b**2 + c**2 - a**2) / (2 * b * c)
        cosB = (c**2 + a**2 - b**2) / (2 * c * a)
        cosC = (a**2 + b**2 - c**2) / (2 * a * b)
        # Convert the cosines to angles in degrees
        angleA = np.arccos(cosA) * 180 / np.pi
        angleB = np.arccos(cosB) * 180 / np.pi
        angleC = np.arccos(cosC) * 180 / np.pi
        
        return angleA, angleB, angleC

    def check_overlap(self, last_triangle_pts, new_triangle_pts):
        # triangle_pts will be : (p1, p2, p3), p1 = (p1y, p1x), ...
        (last_point1, last_point2, last_point3) = last_triangle_pts
        last_point1 = np.array(last_point1)
        last_point2 = np.array(last_point2)
        last_point3 = np.array(last_point3)
        
        a = math.hypot((last_point1 - last_point2)[0], (last_point1 - last_point2)[1])
        b = math.hypot((last_point2 - last_point3)[0], (last_point2 - last_point3)[1])
        c = math.hypot((last_point1 - last_point3)[0], (last_point1 - last_point3)[1])

        s = (a + b + c) / 2
        r = (a * b * c) / (4 * ((s * (s - a) * (s - b) * (s - c)) ** 0.5))
        x0 = (a * last_point1[0] + b * last_point2[0] + c * last_point3[0]) / (a + b + c)
        y0 = (a * last_point1[1] + b * last_point2[1] + c * last_point3[1]) / (a + b + c)
        outcenter = np.array([x0, y0])
        boleanList = []
        for pts in new_triangle_pts:
            pts = np.array(pts)
            len_to_c = math.hypot((pts - outcenter)[0], (pts - outcenter)[1])
            if abs(len_to_c - r) <= 50:
                boleanList.append(True)
            else:
                boleanList.append(False)
        if True in boleanList:
            bool = True
        else:
            bool = False
        return bool
        
    def plot(self):
        for pts in self.vertex_list:
            cv2.drawContours(self.img, [np.array(pts)], 0, (255, 255, 255), 2)    
        cv2.imshow('My Image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
  
    def rotate(self):
        # create initial triangles
        triangles = self.vertex_list
        colors = (255, 255, 255)
        thickness = 2
        for i in range(len(triangles)):
            triangles[i] = np.array(triangles[i], np.int32)

        # set initial angles and rotation speeds for each triangle
        angles = 0
        rotation_speeds = 0.2

        # display animation
        while True:
            # clear previous frame
            self.img[:] = 0

            for i in range(len(triangles)):
                # calculate centroid
                M = cv2.moments(triangles[i])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # rotate triangle around centroid
                angles += rotation_speeds
                R = cv2.getRotationMatrix2D((cx, cy), angles, 1)
                triangle_rotated = cv2.transform(np.array([triangles[i]]), R).astype(np.int32)

                # draw rotated triangle
                color = colors
                cv2.drawContours(self.img, triangle_rotated, 0, color, thickness)

            # display frame and check for key press
            cv2.imshow('Rotating Triangles', self.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
    '''
    def close():
        ...
    '''
img_h = 600
img_w = 800
num_of_triangles = 8
max_area = 1100
min_area = 900
max_degree = 90
min_degree = 30
a = MyClass(img_h, img_w, num_of_triangles, max_area, min_area, max_degree, min_degree)
a.plot()
a.rotate()
#a.close()
