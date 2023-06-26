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
    def __init__(self):
        self.imgsize = (600, 800, 3)
        self.img = np.zeros(self.imgsize,dtype=np.uint8)

        # generate triangle vertex
        self.num_of_triangles = 8
        # conditions
        self.max_area = 1100
        self.min_area = 900
        self.max_degree = 90
        self.min_degree = 30
        self.vertex_list = []
        for i in range(self.num_of_triangles):
            while True:
                p1 = (random.randint(0+50, self.imgsize[1]-50), random.randint(0+50, self.imgsize[0]-50))
                p2 = (random.randint(0+50, self.imgsize[1]-50), random.randint(0+50, self.imgsize[0]-50))
                p3 = (random.randint(0+50, self.imgsize[1]-50), random.randint(0+50, self.imgsize[0]-50))
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
                        if len(self.vertex_list) != 0:
                            # check overlap
                            overlap = False
                            for last_triangle_pts in self.vertex_list:
                                overlap = self.check_overlap(last_triangle_pts, triangle_pts)
                                if overlap:
                                    self.vertex_list.append(triangle_pts)
                                    break
                            if not overlap:
                                self.vertex_list.append(triangle_pts)
                                break
                        else:
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
    '''
    def check_overlap(self, last_triangle_pts, new_triangle_pts):
        # triangle_pts will be : (p1, p2, p3), p1 = (p1y, p1x), ...
        (last_point1, last_point2, last_point3) = last_triangle_pts
        last_point1 = np.array(last_point1)
        last_point2 = np.array(last_point2)
        last_point3 = np.array(last_point3)
        area_last = self.calculate_triangle_area(last_triangle_pts)
        boleanList = []
        for pts in new_triangle_pts:
            pts = np.array(pts)
            triangle_pts_1 = (pts, last_point2, last_point3) 
            area_1 = self.calculate_triangle_area(triangle_pts_1)
            triangle_pts_2 = (last_point1, pts, last_point3) 
            area_2 = self.calculate_triangle_area(triangle_pts_2)
            triangle_pts_3 = (last_point1, last_point2, pts) 
            area_3 = self.calculate_triangle_area(triangle_pts_3)
            area_sum = area_1 + area_2 + area_3
            if abs(area_sum - area_last) <= 1:
                boleanList.append(True)
            else:
                boleanList.append(False)
        return all(boleanList)
    '''
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
            if abs(len_to_c - r) <= 1:
                boleanList.append(True)
            else:
                boleanList.append(False)
        return all(boleanList)
        
    def plot(self):
        for pts in self.vertex_list:
            cv2.drawContours(self.img, [np.array(pts)], 0, (255, 255, 255), -1)    
        cv2.imshow('My Image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
  
    def rotate(self):
        # create initial triangles
        triangles = self.vertex_list
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (128, 128, 128), (255, 255, 255)]
        thickness = 2
        for i in range(8):
            triangles[i] = np.array(triangles[i], np.int32)

        # set initial angles and rotation speeds for each triangle
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        rotation_speeds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        # display animation
        while True:
            # clear previous frame
            self.img[:] = 0

            for i in range(8):
                # calculate centroid
                M = cv2.moments(triangles[i])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # rotate triangle around centroid
                angles[i] += rotation_speeds[i]
                R = cv2.getRotationMatrix2D((cx, cy), angles[i], 1)
                triangle_rotated = cv2.transform(np.array([triangles[i]]), R).astype(np.int32)

                # draw rotated triangle
                color = colors[i]
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
a = MyClass()
a.plot()
a.rotate()
#a.close()
