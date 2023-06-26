import numpy as np
import cv2

class MyHouse():
    
    # Constructor
    # The constructor is a function that automatically
    # runs when you create an object of this class class.
    def __init__(self, width = 200, height = 200):
        self.width = width
        self.height = height
    
    def set_points2d(self):
        self.points2d = np.zeros((4, 2),dtype=float)
        self.points2d[0,0] = 400 - self.width / 2
        self.points2d[0,1] = 400 - self.height / 2 
        self.points2d[1,0] = 400 + self.width / 2 
        self.points2d[1,1] = 400 - self.height / 2  
        self.points2d[2,0] = 400 + self.width / 2 
        self.points2d[2,1] = 400 + self.height / 2  
        self.points2d[3,0] = 400 - self.width / 2 
        self.points2d[3,1] = 400 + self.height / 2  

    def set_line(self):
        self.lines = np.zeros((4,2), dtype=int)
        self.lines[0,0] = 0
        self.lines[0,1] = 1
        self.lines[1,0] = 1
        self.lines[1,1] = 2
        self.lines[2,0] = 2
        self.lines[2,1] = 3
        self.lines[3,0] = 3
        self.lines[3,1] = 0

    def plot_lines(self):
        self.img = np.zeros((800,800,3), dtype=np.uint8)
        n_lines = self.lines.shape[0]
        for i in range(n_lines):
            cv2.line(self.img,
                     self.points2d[self.lines[i,0]].astype('int'),
                     self.points2d[self.lines[i,1]].astype('int'),
                     color=[255,0,0],
                     thickness=2)
    
    def show_image(self):
        cv2.imshow("MyHouse", self.img)
        ikey = cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_image_all_in_one(self):
        self.set_points2d()
        self.set_line()
        self.plot_lines()
        self.show_image()
    
'''        
img = np.zeros((800, 800, 3),dtype=np.uint8)
cv2.line(img, (300, 300), (600, 300), color=(255,0,0),thickness=2)
cv2.line(img, (600, 300), (600, 600), color=(255,0,0),thickness=2)
cv2.line(img, (600, 600), (300, 600), color=(255,0,0),thickness=2)
cv2.line(img, (300, 600), (300, 300), color=(255,0,0),thickness=2)

cv2.imshow("img", img)
ikey = cv2.waitKey(0)
cv2.destroyAllWindows()
'''
obj = MyHouse()
obj.show_image_all_in_one()