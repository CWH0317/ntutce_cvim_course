'''
def triangle_area(x1, y1, x2, y2, x3, y3):
    area = 0.5 * abs((x1 * y2 + x2 * y3 + x3 * y1) - (x2 * y1 + x3 * y2 + x1 * y3))
    return area

my_area = triangle_area(0.0, 0.0, 1.0, 0.0, 0.5, 0.5)
print("The triangle area is ", my_area)
'''
class Triangle:
    def __init__(self):
        x1 = 0.
        y1 = 0.
        x2 = 0.
        y2 = 0.
        x3 = 0.
        y3 = 0.
    
    def setP1(self, a, b):
        self.x1 = a
        self.y1 = b
        
    def setP2(self, a, b):
        self.x2 = a
        self.y2 = b
        
    def setP3(self, a, b):
        self.x3 = a
        self.y3 = b  
    
    def triangle_area(self):
        x1 = self.x1
        y1 = self.y1
        x2 = self.x2
        y2 = self.y2
        x3 = self.x3
        y3 = self.y3
        area = 0.5 * abs((x1 * y2 + x2 * y3 + x3 * y1) - (x2 * y1 + x3 * y2 + x1 * y3))
        return area
    
    def massCenter(self):
        xc = 0.
        yc = 0.
        xc = (self.x1 + self.x2 + self.x3) / 3
        yc = (self.y1 + self.y2 + self.y3) / 3
        return xc, yc

a = Triangle()
a.setP1(0, 0)
a.setP2(1, 0)
a.setP3(0.5, 0.5)

c = a.massCenter()
print("The mass center of the triangle is : " ,c)

area = a.triangle_area()
print("The area of the triangle is : " ,area)