import cv2
import numpy as np

def drawblock(img, h_start, h_end, w_start, w_end):
    img[h_start:h_end, w_start:w_end, 0] = 100
    img[h_start:h_end, w_start:w_end, 1] = 200
    img[h_start:h_end, w_start:w_end, 2] = 255
    
def colorline(img, h_start, h_end, w_start, w_end):
    for j in range(w_start, w_end):
        blue = int((j-w_start)/(w_end-w_start)*255)
        blue = max(min(blue,255),0)
        green = 0
        red = 255 - blue
        img[h_start:h_end,j,0] = blue
        img[h_start:h_end,j,1] = green
        img[h_start:h_end,j,2] = red
        
        
h = 600
w = 800
c = 3

img = np.zeros((h, w, c), dtype=np.uint8)

h_start = 110
h_end = 175
w_start = 155
w_end = 185
colorline(img, h_start, h_end, w_start, w_end)
h_start = 175
h_end = 195
w_start = 75
w_end = 265
colorline(img, h_start, h_end, w_start, w_end)
h_start = 250
h_end = 270
w_start = 105
w_end = 235
colorline(img, h_start, h_end, w_start, w_end)
h_start = 325
h_end = 345
w_start = 105
w_end = 235
colorline(img, h_start, h_end, w_start, w_end)
h_start = 400
h_end = 420
w_start = 95
w_end = 245
colorline(img, h_start, h_end, w_start, w_end)
h_start = 475
h_end = 495
w_start = 95
w_end = 245
colorline(img, h_start, h_end, w_start, w_end)
h_start = 400
h_end = 475
w_start = 95
w_end = 115
colorline(img, h_start, h_end, w_start, w_end)
h_start = 400
h_end = 475
w_start = 225
w_end = 245
colorline(img, h_start, h_end, w_start, w_end)

for i in range(7):
    h_start = 50 + i * 10
    h_end = h_start + 25
    w_start = 360 + i * 10
    w_end = w_start + 25
    colorline(img, h_start, h_end, w_start, w_end)
for i in range(7):
    h_start = 50 + i * 10
    h_end = h_start + 25
    w_start = 580 - i * 10
    w_end = w_start + 25
    colorline(img, h_start, h_end, w_start, w_end) 
     
h_start = 130
h_end = 150
w_start = 300
w_end = 650
colorline(img, h_start, h_end, w_start, w_end)

h_start = 230
h_end = 250
w_start = 350
w_end = 600
colorline(img, h_start, h_end, w_start, w_end)

h_start = 330
h_end = 350
w_start = 300
w_end = 650
colorline(img, h_start, h_end, w_start, w_end)

h_start = 130
h_end = 550
w_start = 470
w_end = 495
colorline(img, h_start, h_end, w_start, w_end)

m = cv2.getRotationMatrix2D((299.5, 399.5), 20, 1)
img2 = cv2.warpAffine(img, m, (w, h))
cv2.imshow("img", img2)
cv2.waitKey(0)

