import cv2 as cv
import numpy as np 

fnames = []
fnames.append(r"D:\Lab\111-2\computer_vision_and_image_measurement\chessboard_practice\189475.jpg")
fnames.append(r"D:\Lab\111-2\computer_vision_and_image_measurement\chessboard_practice\189476.jpg")
fnames.append(r"D:\Lab\111-2\computer_vision_and_image_measurement\chessboard_practice\189477.jpg")

allCorners = np.zeros((3, 70, 2), dtype=float)

for i in range(3):
# Read image
    img = cv.imread(fnames[i], cv.IMREAD_GRAYSCALE)

    # Call findChessboardCorners
    psize = (7, 10)
    ret1 = cv.findChessboardCorners(img, psize)
    if ret1[0] == True:
        corners = ret1[1].reshape(-1, 2)
        allCorners[i, :, :] = corners
    else:
        print("Error: Could not find all corners.")

# Call drawChessboardCorners
    imgWithCorners = img.copy()
    imgWithCorners = cv.cvtColor(imgWithCorners, 
                                 cv.COLOR_GRAY2BGR)
    cv.drawChessboardCorners(imgWithCorners, 
                             psize, 
                             corners,
                             ret1[0])

    # Resize the image 
    myScreen = (1280, 720)
    ratio_x = myScreen[0] / imgWithCorners.shape[1]
    ratio_y = myScreen[1] / imgWithCorners.shape[0]
    ratio = min(ratio_x, ratio_y)
    dsize = (int(ratio * imgWithCorners.shape[1]), 
             int(ratio * imgWithCorners.shape[0]))
    imgWithCornersResized = cv.resize(
        imgWithCorners, dsize)
    
    # display
    winName = "Found corners of image %d" % (i + 1)
    cv.imshow(winName, imgWithCornersResized)
#    ikey = cv.waitKey(0)
    
ikey = cv.waitKey(0)
cv.destroyAllWindows()

# Call cabibrateCamera()
objPoints = np.zeros((3, 70, 3), dtype=float)
dx = 21.27
dy = 21.31
for i in range(10):
    for j in range(7):
        objPoints[0, i * 7 + j, 0] = i * dx
        objPoints[0, i * 7 + j, 1] = j * dy
        objPoints[0, i * 7 + j, 2] = 0.
for i in range(1, 3):
    objPoints[i, :, :] = objPoints[0, :, :]
imgSize = (img.shape[1], img.shape[0])
cmat = np.array([[2000,0,1000],[0,2000,1000],[0,0,1.]])
dvec = np.zeros((1,8))
ret2 = cv.calibrateCamera(objPoints.astype(np.float32), 
                          allCorners.astype(np.float32), imgSize, cmat, dvec)
print(objPoints)
print(allCorners)




